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

package zapcore_test

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"go.uber.org/zap/internal/exit"
	//revive:disable:dot-imports
	. "go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func writeTo(core Core, lvl Level, msg string, fields ...Field) {
	ent := Entry{Level: lvl, Time: time.Now(), Message: msg}
	if ce := core.Check(ent, nil); ce != nil {
		ce.Write(fields...)
	}
}

func newScopedObserver(enab LevelEnabler) (*ScopeCore, *observer.ObservedLogs) {
	outer, logs := observer.New(enab)
	return NewScopeCore(outer), logs
}

func TestScopeCorePassThroughWithoutScope(t *testing.T) {
	sc, logs := newScopedObserver(InfoLevel)

	assert.Equal(t, InfoLevel, LevelOf(sc))
	assert.False(t, sc.Enabled(DebugLevel))
	assert.True(t, sc.Enabled(InfoLevel))

	writeTo(sc, InfoLevel, "one")
	writeTo(sc, DebugLevel, "filtered")
	assert.Equal(t, 1, logs.Len(), "entries must reach the outer Core without an entered scope")
}

func TestScopeCommit(t *testing.T) {
	sc, logs := newScopedObserver(DebugLevel)
	scope := sc.NewScope()

	require.NoError(t, scope.Enter(func() error {
		writeTo(sc, InfoLevel, "buffered-1")
		writeTo(sc, InfoLevel, "buffered-2")
		assert.Equal(t, 0, logs.Len(), "outer Core must not receive entries while the scope is active")
		return nil
	}))

	all := logs.AllUntimed()
	require.Len(t, all, 2)
	assert.Equal(t, "buffered-1", all[0].Message)
	assert.Equal(t, "buffered-2", all[1].Message, "committed entries must keep buffer order")
}

func TestScopeDiscardOnError(t *testing.T) {
	sc, logs := newScopedObserver(DebugLevel)
	scope := sc.NewScope()
	sentinel := errors.New("boom")

	err := scope.Enter(func() error {
		writeTo(sc, InfoLevel, "buffered")
		return sentinel
	})
	assert.ErrorIs(t, err, sentinel)
	assert.Equal(t, 0, logs.Len(), "failed scope must discard its buffer")

	// The scope is reusable after exit, and later writes go out directly.
	writeTo(sc, InfoLevel, "after")
	assert.Equal(t, 1, logs.Len())
}

func TestScopeDiscardOnPanic(t *testing.T) {
	sc, logs := newScopedObserver(DebugLevel)
	scope := sc.NewScope()

	assert.PanicsWithValue(t, "kaboom", func() {
		_ = scope.Enter(func() error {
			writeTo(sc, InfoLevel, "buffered")
			panic("kaboom")
		})
	}, "the panic must propagate after the scope is torn down")
	assert.Equal(t, 0, logs.Len(), "panicked scope must discard its buffer")

	// State must be usable again after the panic.
	require.NoError(t, scope.Enter(func() error {
		writeTo(sc, InfoLevel, "second-scope")
		return nil
	}))
	assert.Equal(t, 1, logs.Len())
}

type mutableLevel struct {
	mu  sync.Mutex
	lvl Level
}

func (m *mutableLevel) Enabled(l Level) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return l >= m.lvl
}

func (m *mutableLevel) set(l Level) {
	m.mu.Lock()
	m.lvl = l
	m.mu.Unlock()
}

func TestScopeLevelSnapshot(t *testing.T) {
	atom := &mutableLevel{lvl: InfoLevel}
	sc, logs := newScopedObserver(atom)
	scope := sc.NewScope()

	require.NoError(t, scope.Enter(func() error {
		// Raising the level after entering must not filter entries that the
		// entry-time snapshot admitted.
		atom.set(ErrorLevel)
		writeTo(sc, InfoLevel, "snapshot-admitted")
		assert.Equal(t, InfoLevel, LevelOf(sc), "active scope reports the snapshot level")

		// Lowering the level after entering must not admit entries that the
		// entry-time snapshot rejected.
		atom.set(DebugLevel)
		writeTo(sc, DebugLevel, "snapshot-rejected")

		// Restore: the active gating decision stays the snapshot throughout.
		atom.set(InfoLevel)
		return nil
	}))

	require.Len(t, logs.All(), 1, "only entries admitted by the entry-time level snapshot may commit")
	assert.Equal(t, "snapshot-admitted", logs.All()[0].Message)

	// After exit, the live outer level is in effect again.
	atom.set(DebugLevel)
	writeTo(sc, DebugLevel, "live-level")
	assert.Equal(t, 2, logs.Len())
}

func TestScopeRetainsWithFields(t *testing.T) {
	sc, logs := newScopedObserver(DebugLevel)

	// Child logger created BEFORE the scope is entered.
	child := sc.With([]Field{strField("pre", "yes")})

	scope := sc.NewScope()
	require.NoError(t, scope.Enter(func() error {
		writeTo(child, InfoLevel, "from-child")
		writeTo(child, InfoLevel, "from-child-site", strField("site", "ok"))
		return nil
	}))

	all := logs.AllUntimed()
	require.Len(t, all, 2)
	assert.Equal(t, map[string]string{"pre": "yes"}, fieldsMap(all[0].Context))
	assert.Equal(t, map[string]string{"pre": "yes", "site": "ok"}, fieldsMap(all[1].Context))

	// Fields of siblings stay isolated.
	sibling := sc.With([]Field{strField("sibling", "1")})
	writeTo(sibling, InfoLevel, "direct")
	last := logs.AllUntimed()[2]
	assert.Equal(t, map[string]string{"sibling": "1"}, fieldsMap(last.Context))
}

func strField(key, val string) Field {
	return Field{Key: key, Type: StringType, String: val}
}

func TestScopeChildAfterExitWritesOuter(t *testing.T) {
	sc, logs := newScopedObserver(DebugLevel)
	child := sc.With([]Field{strField("k", "v")})
	scope := sc.NewScope()

	require.NoError(t, scope.Enter(func() error {
		writeTo(child, InfoLevel, "buffered")
		return nil
	}))
	require.Len(t, logs.All(), 1)

	writeTo(child, InfoLevel, "direct-after-exit")
	require.Len(t, logs.All(), 2, "child loggers must not keep writing to a closed buffer")
	assert.Equal(t, "direct-after-exit", logs.All()[1].Message)
}

func TestScopeNestedCommit(t *testing.T) {
	sc, logs := newScopedObserver(DebugLevel)
	outer := sc.NewScope()
	inner := sc.NewScope()

	require.NoError(t, outer.Enter(func() error {
		writeTo(sc, InfoLevel, "outer-before")
		require.NoError(t, inner.Enter(func() error {
			writeTo(sc, InfoLevel, "inner")
			assert.Equal(t, 0, logs.Len(), "inner scope commits to the outer buffer, not the outer Core")
			return nil
		}))
		writeTo(sc, InfoLevel, "outer-after")
		assert.Equal(t, 0, logs.Len(), "outer Core sees nothing until the outer scope commits")
		return nil
	}))

	msgs := messages(logs.All())
	assert.Equal(t, []string{"outer-before", "inner", "outer-after"}, msgs)
}

func TestScopeNestedFailureIsInvisible(t *testing.T) {
	sc, logs := newScopedObserver(DebugLevel)
	outer := sc.NewScope()
	inner := sc.NewScope()

	require.NoError(t, outer.Enter(func() error {
		writeTo(sc, InfoLevel, "outer-before")
		err := inner.Enter(func() error {
			writeTo(sc, InfoLevel, "inner-failed")
			return errors.New("nope")
		})
		assert.Error(t, err)
		writeTo(sc, InfoLevel, "outer-after")
		return nil
	}))

	assert.Equal(t, []string{"outer-before", "outer-after"}, messages(logs.All()))
}

func TestScopeNestedPanicDiscardsBoth(t *testing.T) {
	sc, logs := newScopedObserver(DebugLevel)
	outer := sc.NewScope()
	inner := sc.NewScope()

	assert.PanicsWithValue(t, "inner-boom", func() {
		_ = outer.Enter(func() error {
			writeTo(sc, InfoLevel, "outer")
			_ = inner.Enter(func() error {
				writeTo(sc, InfoLevel, "inner")
				panic("inner-boom")
			})
			return nil
		})
	})
	assert.Equal(t, 0, logs.Len(), "an inner panic discards the outer buffer too")
}

func TestScopeReentrySameScope(t *testing.T) {
	sc, _ := newScopedObserver(DebugLevel)
	scope := sc.NewScope()

	require.NoError(t, scope.Enter(func() error {
		// Same goroutine, scope not exited yet.
		assert.ErrorIs(t, scope.Enter(func() error { return nil }), ErrScopeActive)

		// Different goroutine.
		var wg sync.WaitGroup
		wg.Add(1)
		var err2 error
		go func() {
			defer wg.Done()
			err2 = scope.Enter(func() error { return nil })
		}()
		wg.Wait()
		assert.ErrorIs(t, err2, ErrScopeActive)
		return nil
	}))
}

func TestScopeConcurrentFamilyBusy(t *testing.T) {
	sc, _ := newScopedObserver(DebugLevel)
	first := sc.NewScope()
	second := sc.NewScope()

	var wg sync.WaitGroup
	var err error

	require.NoError(t, first.Enter(func() error {
		wg.Add(1)
		go func() {
			defer wg.Done()
			err = second.Enter(func() error { return nil })
		}()
		wg.Wait()
		assert.ErrorIs(t, err, ErrScopeBusy,
			"a different goroutine must not enter the active family")
		return nil
	}))
}

func TestScopeParallelFamilies(t *testing.T) {
	sc1, logs1 := newScopedObserver(DebugLevel)
	sc2, logs2 := newScopedObserver(DebugLevel)

	var wg sync.WaitGroup
	run := func(sc *ScopeCore, logs *observer.ObservedLogs) {
		defer wg.Done()
		err := sc.NewScope().Enter(func() error {
			for i := 0; i < 50; i++ {
				writeTo(sc, InfoLevel, "x")
			}
			assert.Equal(t, 0, logs.Len())
			return nil
		})
		require.NoError(t, err)
	}
	wg.Add(2)
	go run(sc1, logs1)
	go run(sc2, logs2)
	wg.Wait()

	assert.Equal(t, 50, logs1.Len())
	assert.Equal(t, 50, logs2.Len(), "parallel scope families must not interfere")
}

func TestScopeSamplingAtCommit(t *testing.T) {
	outer, logs := observer.New(DebugLevel)
	var sampled, dropped atomic.Int64
	sampledCore := NewSamplerWithOptions(outer, time.Minute, 2, 3, SamplerHook(func(_ Entry, dec SamplingDecision) {
		if dec&LogSampled != 0 {
			sampled.Add(1)
		}
		if dec&LogDropped != 0 {
			dropped.Add(1)
		}
	}))
	sc := NewScopeCore(sampledCore)
	scope := sc.NewScope()

	require.NoError(t, scope.Enter(func() error {
		for i := 0; i < 10; i++ {
			writeTo(sc, InfoLevel, "same")
		}
		assert.Equal(t, int64(0), sampled.Load()+dropped.Load(),
			"outer sampler must not count buffered entries before commit")
		return nil
	}))

	// Per-entry counting at commit: 10 decisions, sampler allows 1,2,5,8.
	assert.Equal(t, int64(10), sampled.Load()+dropped.Load())
	assert.Equal(t, 4, logs.Len())
}

func TestScopeSamplingDropsAtCommit(t *testing.T) {
	outer, logs := observer.New(DebugLevel)
	sampledCore := NewSamplerWithOptions(outer, time.Minute, 1, 0) // first only
	sc := NewScopeCore(sampledCore)

	require.NoError(t, sc.NewScope().Enter(func() error {
		for i := 0; i < 5; i++ {
			writeTo(sc, InfoLevel, "same")
		}
		return nil
	}))
	assert.Equal(t, 1, logs.Len(), "outer sampler applies to the replayed entries")
}

func TestScopeSideBySideSamplingIsolation(t *testing.T) {
	outer, logs := observer.New(DebugLevel)
	var decisions atomic.Int64
	sampledCore := NewSamplerWithOptions(outer, time.Minute, 100, 0, SamplerHook(func(_ Entry, dec SamplingDecision) {
		decisions.Add(1)
	}))
	sc := NewScopeCore(sampledCore)

	// First scope: buffering must not consume sampling budget.
	require.NoError(t, sc.NewScope().Enter(func() error {
		for i := 0; i < 10; i++ {
			writeTo(sc, InfoLevel, "msg")
		}
		assert.Equal(t, int64(0), decisions.Load())
		return nil
	}))
	assert.Equal(t, int64(10), decisions.Load())
	assert.Equal(t, 10, logs.Len())

	// A sibling scope starts from untouched counters of its own buffer; only
	// the outer sampler at its own commit counts.
	require.NoError(t, sc.NewScope().Enter(func() error {
		for i := 0; i < 10; i++ {
			writeTo(sc, InfoLevel, "msg")
		}
		assert.Equal(t, int64(10), decisions.Load(), "no cross-talk while the second scope buffers")
		return nil
	}))
	assert.Equal(t, int64(20), decisions.Load())
	assert.Equal(t, 20, logs.Len())
}

func TestScopeFatalDiscardsWithoutFlush(t *testing.T) {
	sc, logs := newScopedObserver(DebugLevel)
	scope := sc.NewScope()

	stub := exit.Stub()
	err := scope.Enter(func() error {
		writeTo(sc, InfoLevel, "before-fatal")
		// Simulate Logger.Fatal: Check/Write at FatalLevel, then the
		// WriteThenFatal terminal hook (stubbed here so it returns normally).
		writeTo(sc, FatalLevel, "fatal")
		exit.With(1)
		return nil
	})
	stub.Unstub()

	require.NoError(t, err)
	assert.True(t, stub.Exited)
	assert.Equal(t, 0, logs.Len(), "buffer must be discarded on the Fatal path, never flushed")
}

func TestScopePanicLevelMarksTerminal(t *testing.T) {
	sc, logs := newScopedObserver(FatalLevel) // PanicLevel disabled
	scope := sc.NewScope()

	stub := exit.Stub()
	err := scope.Enter(func() error {
		writeTo(sc, InfoLevel, "info")
		// A disabled Panic entry is not buffered, but terminal behavior is
		// still in effect: the buffer must not commit afterwards.
		writeTo(sc, PanicLevel, "panic")
		exit.With(1)
		return nil
	})
	stub.Unstub()

	require.NoError(t, err)
	assert.Equal(t, 0, logs.Len())
}

// recordingCore records write order to verify tee fan-out at replay.
type recordingCore struct {
	name string
	log  *[]string
	mu   *sync.Mutex
	enab LevelEnabler
}

func (c *recordingCore) Enabled(l Level) bool { return c.enab.Enabled(l) }
func (c *recordingCore) With([]Field) Core    { return c }
func (c *recordingCore) Sync() error          { return nil }
func (c *recordingCore) Check(e Entry, ce *CheckedEntry) *CheckedEntry {
	if c.Enabled(e.Level) {
		return ce.AddCore(e, c)
	}
	return ce
}

func (c *recordingCore) Write(e Entry, _ []Field) error {
	c.mu.Lock()
	*c.log = append(*c.log, c.name+":"+e.Message)
	c.mu.Unlock()
	return nil
}

func TestScopeCommitFollowsTeeOrder(t *testing.T) {
	var mu sync.Mutex
	var got []string
	a := &recordingCore{name: "a", log: &got, mu: &mu, enab: DebugLevel}
	b := &recordingCore{name: "b", log: &got, mu: &mu, enab: DebugLevel}
	sc := NewScopeCore(NewTee(a, b))

	require.NoError(t, sc.NewScope().Enter(func() error {
		writeTo(sc, InfoLevel, "1")
		writeTo(sc, InfoLevel, "2")
		mu.Lock()
		assert.Empty(t, got)
		mu.Unlock()
		return nil
	}))

	assert.Equal(t, []string{"a:1", "b:1", "a:2", "b:2"}, got)
}

func TestScopePreservesEntry(t *testing.T) {
	sc, logs := newScopedObserver(DebugLevel)
	when := time.Date(2026, 1, 2, 3, 4, 5, 6, time.UTC)
	name := "named"

	require.NoError(t, sc.NewScope().Enter(func() error {
		ce := sc.Check(Entry{Level: InfoLevel, Time: when, LoggerName: name, Message: "m", Caller: EntryCaller{Defined: true, File: "f.go", Line: 7}}, nil)
		require.NotNil(t, ce)
		ce.Write()
		return nil
	}))

	all := logs.All()
	require.Len(t, all, 1)
	assert.Equal(t, when, all[0].Time, "original timestamp must survive buffering (sampler windows depend on it)")
	assert.Equal(t, name, all[0].LoggerName)
	assert.Equal(t, "f.go", all[0].Caller.File)
}

func fieldsMap(fields []Field) map[string]string {
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
