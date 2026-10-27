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

	"go.uber.org/zap/zapcore"
)

var (
	errObservationUnavailable = errors.New(
		"zap: logger was not built with zap.WithObservation; cannot enter an observation scope",
	)
	errObservationNoFunction = errors.New(
		"zap: observation scope requires a non-nil function",
	)
)

// WithObservation installs an observation buffering scope Core around the
// Logger's current Core. Loggers built with this option (and child loggers
// derived from them via With, Named, WithLazy or Sugar) can enter explicit
// observation scopes with [Logger.Observation].
//
// Without an entered scope the installed Core is a transparent pass-through,
// so Level, Sampling, Caller, Stack, Fatal/Panic hooks and all other behavior
// stay unchanged. See [zapcore.NewScopeCore] for the full buffering and
// sampling semantics.
func WithObservation() Option {
	return optionFunc(func(log *Logger) {
		if log.observation != nil {
			return
		}
		sc := zapcore.NewScopeCore(log.core)
		log.core = sc
		log.observation = sc
	})
}

// Observation returns a handle for entering an observation buffering scope on
// this Logger's Core family, or nil if the Logger was not built with
// [WithObservation] (or constructed around a [zapcore.ScopeCore]).
//
// The handle is cheap: call it once and keep it, or call it again inside a
// running scope to obtain a fresh handle for a nested scope. Every handle is
// independently enterable, but the underlying family admits only one
// top-level scope at a time, entered by one goroutine.
func (log *Logger) Observation() *Observation {
	sc := log.observation
	if sc == nil {
		// Allow loggers constructed directly around a *zapcore.ScopeCore.
		if core, ok := log.core.(*zapcore.ScopeCore); ok {
			sc = core
		}
	}
	if sc == nil {
		return nil
	}
	return &Observation{scope: sc.NewScope(), log: log}
}

// Observation enters explicit observation buffering scopes on a Logger family
// enabled with [WithObservation].
//
// An Observation value must not be shared with Enter concurrently; obtain
// separate values (via [Logger.Observation]) for parallel or nested scopes.
type Observation struct {
	scope *zapcore.Scope
	log   *Logger
}

// Enter runs fn inside an observation scope, passing the Logger the scope is
// bound to as its argument.
//
// While fn runs, entries written through the Logger and any of its child
// loggers (including children created before the scope was entered) are held
// in the scope buffer. If fn returns nil, the entries are committed to the
// outer Core chain through its ordinary Check/Write path; the outer sampling
// Core, if any, makes one sampling decision per entry at commit time using the
// entry's original timestamp. If fn returns a non-nil error, the buffer is
// discarded and that error is returned. A panic raised inside fn (including a
// Panic-level log, or development DPanic) discards the buffer and is
// re-raised; a Fatal-level log discards the buffer and terminates the process
// without ever flushing it.
//
// Enter returns [zapcore.ErrScopeActive] when this scope is already entered,
// or [zapcore.ErrScopeBusy] when the family is entered in another goroutine.
func (o *Observation) Enter(fn func(*Logger) error) error {
	if o == nil || o.scope == nil {
		return errObservationUnavailable
	}
	if fn == nil {
		return errObservationNoFunction
	}
	return o.scope.Enter(func() error { return fn(o.log) })
}
