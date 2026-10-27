// Copyright (c) 2026 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package zap

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"go.uber.org/zap/internal/exit"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func withObservation(t testing.TB, enab zapcore.LevelEnabler, extra ...Option) (*Logger, *observer.ObservedLogs) {
	core, logs := observer.New(enab)
	opts := append([]Option{WithObservation()}, extra...)
	return New(core, opts...), logs
}

func TestObservationRequiresOption(t *testing.T) {
	core, _ := observer.New(DebugLevel)
	log := New(core)
	assert.Nil(t, log.Observation(), "logger without WithObservation has no scope handle")
	assert.Error(t, (*Observation)(nil).Enter(func(*Logger) error { return nil }))
}

func TestObservationPassThrough(t *testing.T) {
	log, logs := withObservation(t, InfoLevel)
	log.Info("one")
	log.Debug("filtered")
	assert.Equal(t, 1, logs.Len(), "no entered scope: behavior is unchanged")
}

func TestObservationCommit(t *testing.T) {
	log, logs := withObservation(t, DebugLevel)
	child := log.Named("worker").With(String("req", "42"))

	require.NoError(t, log.Observation().Enter(func(in *Logger) error {
		in.Info("from-scoped")
		child.Info("from-child")
		assert.Equal(t, 0, logs.Len())
		return nil
	}))

	all := logs.AllUntimed()
	require.Len(t, all, 2)
	assert.Equal(t, "from-scoped", all[0].Message)
	assert.Equal(t, "from-child", all[1].Message)
	assert.Equal(t, "worker", all[1].LoggerName)
	require.Len(t, all[1].Context, 1)
	assert.True(t, all[1].Context[0].Equals(String("req", "42")))

	// Child loggers revert to the outer Core after the scope exits.
	child.Info("after")
	assert.Equal(t, 3, logs.Len())
}

func TestObservationChildCreatedInside(t *testing.T) {
	log, logs := withObservation(t, DebugLevel)
	require.NoError(t, log.Observation().Enter(func(in *Logger) error {
		l := in.With(String("k", "v"))
		l.Info("inner-child")
		return nil
	}))
	all := logs.AllUntimed()
	require.Len(t, all, 1)
	assert.Equal(t, map[string]string{"k": "v"}, fieldMap(all[0].Context))
}

func TestObservationDiscardOnError(t *testing.T) {
	log, logs := withObservation(t, DebugLevel)
	sentinel := errors.New("nope")

	err := log.Observation().Enter(func(in *Logger) error {
		in.Info("dropped")
		return sentinel
	})
	assert.ErrorIs(t, err, sentinel)
	assert.Equal(t, 0, logs.Len())

	log.Info("after")
	assert.Equal(t, 1, logs.Len())
}

func TestObservationDiscardOnPanic(t *testing.T) {
	log, logs := withObservation(t, DebugLevel)
	assert.PanicsWithValue(t, "boom", func() {
		_ = log.Observation().Enter(func(in *Logger) error {
			in.Info("dropped")
			panic("boom")
		})
	})
	assert.Equal(t, 0, logs.Len())
}

func TestObservationPanicLevelDiscards(t *testing.T) {
	log, logs := withObservation(t, DebugLevel)
	assert.Panics(t, func() {
		_ = log.Observation().Enter(func(in *Logger) error {
			in.Info("before")
			in.Panic("terminal-panic")
			return nil
		})
	})
	assert.Equal(t, 0, logs.Len(), "Panic entry and the whole buffer are discarded")
}

func TestObservationFatalDiscards(t *testing.T) {
	log, logs := withObservation(t, DebugLevel)
	stub := exit.WithStub(func() {
		err := log.Observation().Enter(func(in *Logger) error {
			in.Info("before")
			in.Fatal("terminal-fatal")
			return nil
		})
		require.NoError(t, err)
	})
	assert.True(t, stub.Exited)
	assert.Equal(t, 0, logs.Len(), "Fatal must never flush the buffered entries")
}

func TestObservationFatalGoexitDiscards(t *testing.T) {
	// A Fatal hook that ends in runtime.Goexit still runs deferred funcs, so
	// the scope must discard rather than commit.
	core, logs := observer.New(DebugLevel)
	log := New(core, WithObservation(), WithFatalHook(zapcore.WriteThenGoexit))

	done := make(chan struct{})
	var enterErr error
	go func() {
		defer close(done)
		enterErr = log.Observation().Enter(func(in *Logger) error {
			in.Info("before")
			in.Fatal("goexit-fatal")
			return nil
		})
	}()
	<-done
	require.NoError(t, enterErr)
	assert.Equal(t, 0, logs.Len())
}

func TestObservationDPanicDevelopmentDiscards(t *testing.T) {
	log, logs := withObservation(t, DebugLevel, Development())
	assert.Panics(t, func() {
		_ = log.Observation().Enter(func(in *Logger) error {
			in.DPanic("dev-dpanic")
			return nil
		})
	})
	assert.Equal(t, 0, logs.Len())
}

func TestObservationLevelSnapshot(t *testing.T) {
	atom := NewAtomicLevelAt(InfoLevel)
	core, logs := observer.New(atom)
	log := New(core, WithObservation())

	require.NoError(t, log.Observation().Enter(func(in *Logger) error {
		atom.SetLevel(ErrorLevel)
		// In-scope gating uses the entry-time snapshot: Info is admitted and
		// Debug (which would pass under a lowered level) never enters.
		in.Info("admitted-by-snapshot")
		in.Debug("not-admitted-at-entry")
		atom.SetLevel(InfoLevel)
		return nil
	}))
	require.Len(t, logs.All(), 1)
	assert.Equal(t, "admitted-by-snapshot", logs.All()[0].Message)
}

func TestObservationNested(t *testing.T) {
	log, logs := withObservation(t, DebugLevel)
	outer := log.Observation()
	require.NoError(t, outer.Enter(func(in *Logger) error {
		in.Info("outer-a")
		// Fresh handle for the nested scope.
		inner := in.Observation()
		require.NoError(t, inner.Enter(func(deep *Logger) error {
			deep.Info("inner")
			return nil
		}))
		assert.Equal(t, 0, logs.Len(), "inner commit stays inside the outer buffer")
		in.Info("outer-b")
		return nil
	}))
	assert.Equal(t,
		[]string{"outer-a", "inner", "outer-b"},
		messages(logs.AllUntimed()))
}

func TestObservationNestedFailureHidden(t *testing.T) {
	log, logs := withObservation(t, DebugLevel)
	require.NoError(t, log.Observation().Enter(func(in *Logger) error {
		in.Info("outer-a")
		err := in.Observation().Enter(func(deep *Logger) error {
			deep.Info("inner-failed")
			return errors.New("x")
		})
		assert.Error(t, err)
		in.Info("outer-b")
		return nil
	}))
	assert.Equal(t, []string{"outer-a", "outer-b"}, messages(logs.AllUntimed()))
}

func TestObservationReentry(t *testing.T) {
	log, _ := withObservation(t, DebugLevel)
	scope := log.Observation()
	require.NoError(t, scope.Enter(func(in *Logger) error {
		assert.ErrorIs(t, scope.Enter(func(*Logger) error { return nil }),
			zapcore.ErrScopeActive)
		return nil
	}))
}

func TestObservationConcurrentBusy(t *testing.T) {
	log, _ := withObservation(t, DebugLevel)
	first := log.Observation()
	second := log.Observation()
	var wg sync.WaitGroup
	require.NoError(t, first.Enter(func(in *Logger) error {
		wg.Add(1)
		go func() {
			defer wg.Done()
			err := second.Enter(func(*Logger) error { return nil })
			assert.ErrorIs(t, err, zapcore.ErrScopeBusy)
		}()
		wg.Wait()
		return nil
	}))
}

func TestObservationSamplingAtCommit(t *testing.T) {
	sink, logs := observer.New(DebugLevel)
	var sampled, dropped atomic.Int64
	sampler := zapcore.NewSamplerWithOptions(sink, time.Minute, 2, 3,
		zapcore.SamplerHook(func(_ zapcore.Entry, dec zapcore.SamplingDecision) {
			if dec&zapcore.LogSampled != 0 {
				sampled.Add(1)
			}
			if dec&zapcore.LogDropped != 0 {
				dropped.Add(1)
			}
		}))
	log := New(sampler, WithObservation())

	require.NoError(t, log.Observation().Enter(func(in *Logger) error {
		for i := 0; i < 10; i++ {
			in.Info("same")
		}
		assert.Equal(t, int64(0), sampled.Load()+dropped.Load())
		return nil
	}))
	assert.Equal(t, int64(10), sampled.Load()+dropped.Load(),
		"outer sampler counts each committed entry once, at commit time")
	assert.Equal(t, 4, logs.Len(), "first 2, then every 3rd: 1,2,5,8")
}

func TestObservationSugar(t *testing.T) {
	log, logs := withObservation(t, DebugLevel)
	sugar := log.Sugar()
	child := sugar.With("pre", "yes")

	require.NoError(t, log.Observation().Enter(func(in *Logger) error {
		child.Infow("from-sugar", "site", "ok")
		return nil
	}))
	all := logs.AllUntimed()
	require.Len(t, all, 1)
	assert.Equal(t, map[string]string{"pre": "yes", "site": "ok"}, fieldMap(all[0].Context))
}

func TestObservationWithLazy(t *testing.T) {
	log, logs := withObservation(t, DebugLevel)
	child := log.WithLazy(String("lazy", "val"))
	require.NoError(t, log.Observation().Enter(func(in *Logger) error {
		child.Info("buffered")
		return nil
	}))
	all := logs.AllUntimed()
	require.Len(t, all, 1)
	assert.Equal(t, map[string]string{"lazy": "val"}, fieldMap(all[0].Context))
}

func TestObservationPreservesCaller(t *testing.T) {
	log, logs := withObservation(t, DebugLevel, AddCaller())
	require.NoError(t, log.Observation().Enter(func(in *Logger) error {
		in.Info("with-caller")
		return nil
	}))
	require.Len(t, logs.All(), 1)
	assert.True(t, logs.All()[0].Caller.Defined)
	assert.Contains(t, logs.All()[0].Caller.File, "observation_test.go")
}

func fieldMap(fields []zapcore.Field) map[string]string {
	m := make(map[string]string, len(fields))
	for _, f := range fields {
		m[f.Key] = f.String
	}
	return m
}

func messages(entries []observer.LoggedEntry) []string {
	out := make([]string, len(entries))
	for i, e := range entries {
		out[i] = e.Message
	}
	return out
}
