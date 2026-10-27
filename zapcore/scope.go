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

package zapcore

import (
	"errors"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
)

// Observation scope sentinel errors.
var (
	// ErrScopeActive is returned when a Scope is entered while it has not
	// been exited yet, whether by the same goroutine or a different one.
	ErrScopeActive = errors.New("zapcore: observation scope is already active")

	// ErrScopeBusy is returned when a Scope is entered while the same
	// ScopeCore family is currently inside another scope in a different
	// goroutine. A family of loggers can only host one top-level scope at a
	// time; nested scopes must be entered from the goroutine that owns the
	// enclosing scope.
	ErrScopeBusy = errors.New("zapcore: observation scope is active in another goroutine")
)

// NewScopeCore wraps outer in a Core that supports explicitly entered
// observation buffering scopes.
//
// When no scope is entered, the returned Core (and every Core derived from it
// via With) is a transparent pass-through to outer: Enabled, Check, Write and
// Sync behave exactly like outer does.
//
// A scope is entered through a handle created with [ScopeCore.NewScope]. While
// a scope is entered, every Check/Write made through the ScopeCore family is
// diverted into the scope's in-memory buffer instead of reaching outer:
//
//   - The level decision uses a snapshot of outer's Enabled answers taken when
//     the scope was entered; entries that fail the level snapshot never enter
//     the buffer.
//   - Fields accumulated on the writing logger via With, as well as fields
//     supplied at the log site, are retained together with the entry.
//   - Nothing reaches outer, including outer's sampling counters.
//
// When the scope exits successfully, the buffered entries are replayed to
// outer once each and in buffered order via the ordinary Check/Write path:
// outer.Check(ent, nil) followed by CheckedEntry.Write(fields...). This means
// an outer sampling Core makes its sampling decision at commit time, exactly
// once per committed entry (using the entry's original timestamp), and may
// drop entries then. This is the documented sampling contract for buffered
// entries; entries written without an entering scope are sampled immediately
// as usual.
//
// When the scope exits because the callback returned an error or panicked
// (including Panic/DPanic/Fatal terminal hooks), the buffer is discarded and
// outer receives none of its entries.
//
// Scopes nest when entered from the goroutine that owns the enclosing scope:
// an inner scope commits into the enclosing scope's buffer rather than into
// outer, and a failed inner scope leaves no trace in the enclosing buffer.
//
// Fields added to a logger with With before a scope is entered are retained on
// the ScopeCore family, so such child loggers automatically write into an
// active scope's buffer; after the scope exits they write to outer again.
func NewScopeCore(outer Core) *ScopeCore {
	if outer == nil {
		outer = NewNopCore()
	}
	return &ScopeCore{
		outer:     outer,
		withOuter: outer,
		state:     &scopeState{},
	}
}

// ScopeCore is a Core that routes writes into an active observation scope's
// buffer, and passes writes through to its outer Core otherwise.
//
// ScopeCore is safe for concurrent use. Do not swap it for another Core while
// a scope is active; scoping is a routing behavior of this Core, not a
// process-global logger replacement.
type ScopeCore struct {
	// outer is the pristine family Core; withOuter is outer with this
	// derivative's With fields already applied (equal to outer when there
	// are none). Active scopes replay into outer, passing the buffered
	// fields per entry; inactive writes behave exactly like a plain logger
	// and go through withOuter, preserving eager With evaluation.
	outer     Core
	withOuter Core
	state     *scopeState
	fields    []Field
}

var (
	_ Core           = (*ScopeCore)(nil)
	_ leveledEnabler = (*ScopeCore)(nil)
)

// scopeState is shared by every ScopeCore derived from the same family via
// With. head is the innermost active buffer, or nil when the family is not
// inside a scope.
type scopeState struct {
	// mu serializes scope enter/exit and the owner bookkeeping. Entry
	// appends do not take it; they synchronize on the buffer's own mutex.
	mu    sync.Mutex
	head  atomic.Pointer[scopeBuffer]
	owner int64 // goroutine ID owning the top-level scope; 0 when inactive
}

// scopeBuffer is one entered scope's buffer. parent links to the enclosing
// scope's buffer (nil for a top-level scope).
type scopeBuffer struct {
	parent  *scopeBuffer
	mu      sync.Mutex
	entries []bufferedEntry
	levels  []bool // len _numLevels, snapshot of outer.Enabled at entry
	// terminal is set when a Panic/Fatal-level entry is seen (even if it is
	// filtered out by the level snapshot), so the buffer is discarded rather
	// than flushed ahead of process termination.
	terminal bool
}

type bufferedEntry struct {
	ent    Entry
	fields []Field
}

// scopeAppender is the Core registered on a CheckedEntry while a scope is
// active; its Write appends the entry to the scope buffer instead of touching
// outer.
type scopeAppender struct {
	buffer *scopeBuffer
	fields []Field
}

var _ Core = (*scopeAppender)(nil)

// NewScope returns an independently enterable scope handle bound to this
// ScopeCore family.
//
// A Scope may be entered again after it has been exited, but it must not be
// entered concurrently from two goroutines, or re-entered from the same
// goroutine before it exits; Enter returns an error in both cases. Use
// distinct handles for nested scopes.
func (c *ScopeCore) NewScope() *Scope {
	return &Scope{core: c}
}

// Enabled reports whether an entry at lvl would be accepted. Inside an active
// scope this uses the level snapshot captured when the innermost scope was
// entered; outside a scope it delegates to the outer Core.
func (c *ScopeCore) Enabled(lvl Level) bool {
	b := c.state.head.Load()
	if b == nil {
		return c.outer.Enabled(lvl)
	}
	return b.allows(lvl, c.outer)
}

// Level reports the minimum enabled level, honoring an active scope's level
// snapshot.
func (c *ScopeCore) Level() Level {
	b := c.state.head.Load()
	if b == nil {
		return LevelOf(c.outer)
	}
	for lvl := _minLevel; lvl <= _maxLevel; lvl++ {
		if b.levels[int(lvl-_minLevel)] {
			return lvl
		}
	}
	return InvalidLevel
}

// With returns a family Core that carries the supplied structured context.
// The derived Core shares the scope state, so loggers created before a scope
// is entered write into the scope buffer once it is active.
func (c *ScopeCore) With(fields []Field) Core {
	return &ScopeCore{
		outer:     c.outer,
		withOuter: c.withOuter.With(fields),
		state:     c.state,
		fields:    combineFields(c.fields, fields),
	}
}

// Check adds the active scope's buffering appender when a scope is entered and
// the entry passes its level snapshot; otherwise it delegates to the outer
// Core with this derivative's With fields already applied.
func (c *ScopeCore) Check(ent Entry, ce *CheckedEntry) *CheckedEntry {
	b := c.state.head.Load()
	if b == nil {
		return c.withOuter.Check(ent, ce)
	}

	// Panic/Fatal terminal behavior is unconditional in Logger.check, even
	// when the level is disabled, so mark the buffer terminal regardless of
	// the level decision.
	if ent.Level >= PanicLevel {
		b.mu.Lock()
		b.terminal = true
		b.mu.Unlock()
	}
	if !b.allows(ent.Level, c.outer) {
		return ce
	}
	return ce.AddCore(ent, &scopeAppender{buffer: b, fields: c.fields})
}

// Write delegates to the outer Core. It is only reached when a ScopeCore is
// registered on a CheckedEntry directly (which never happens while a scope is
// active).
func (c *ScopeCore) Write(ent Entry, fields []Field) error {
	return c.outer.Write(ent, fields)
}

// Sync delegates to the outer Core while no scope is active. While a scope is
// active, it is a no-op: buffered entries have not reached outer, and a scope
// must not interact with the outer Core until it exits.
func (c *ScopeCore) Sync() error {
	if c.state.head.Load() != nil {
		return nil
	}
	return c.outer.Sync()
}

func (app *scopeAppender) Enabled(Level) bool { return true }

func (app *scopeAppender) With([]Field) Core { return app }

func (app *scopeAppender) Check(ent Entry, ce *CheckedEntry) *CheckedEntry {
	return ce.AddCore(ent, app)
}

func (app *scopeAppender) Write(ent Entry, fields []Field) error {
	app.buffer.append(ent, app.fields, fields)
	return nil
}

func (app *scopeAppender) Sync() error { return nil }

func (b *scopeBuffer) allows(lvl Level, outer Core) bool {
	if lvl < _minLevel || lvl > _maxLevel {
		// Levels outside zap's standard set were not covered by the entry
		// snapshot; fall back to the outer Core's current decision.
		return outer.Enabled(lvl)
	}
	return b.levels[int(lvl-_minLevel)]
}

func (b *scopeBuffer) append(ent Entry, context, site []Field) {
	var fields []Field
	if n := len(context) + len(site); n > 0 {
		fields = make([]Field, 0, n)
		fields = append(fields, context...)
		fields = append(fields, site...)
	}
	b.mu.Lock()
	b.entries = append(b.entries, bufferedEntry{ent: ent, fields: fields})
	b.mu.Unlock()
}

func snapshotLevels(outer Core) []bool {
	levels := make([]bool, _numLevels)
	for i := range levels {
		levels[i] = outer.Enabled(_minLevel + Level(i))
	}
	return levels
}

func combineFields(a, b []Field) []Field {
	if len(b) == 0 {
		return a
	}
	out := make([]Field, 0, len(a)+len(b))
	out = append(out, a...)
	out = append(out, b...)
	return out
}

// Scope is a single-use-at-a-time handle for entering an observation
// buffering scope on a [ScopeCore] family.
//
// A Scope is safe for Enter/Exit bookkeeping from one goroutine at a time;
// concurrent or re-entrant Enter calls return an error.
type Scope struct {
	core *ScopeCore

	// The following fields are guarded by core.state.mu.
	active bool
	buffer *scopeBuffer
}

// Enter runs fn inside the observation scope.
//
// While fn runs, writes through the ScopeCore family are buffered. If fn
// returns nil, the buffer is committed: nested scopes hand their entries to
// the enclosing buffer, a top-level scope replays its entries to the outer
// Core through the regular Check/Write path (see [NewScopeCore] for the
// sampling contract). If fn returns an error, the buffer is discarded and
// that error is returned. If fn panics, the buffer is discarded and the panic
// is re-raised after the scope has been torn down; a Fatal hook that calls
// os.Exit prevents the process from reaching the commit point entirely.
//
// Enter returns [ErrScopeActive] if this Scope is already entered, or
// [ErrScopeBusy] if another goroutine currently owns the family.
func (s *Scope) Enter(fn func() error) (retErr error) {
	if fn == nil {
		return errors.New("zapcore: observation scope requires a non-nil function")
	}

	st := s.core.state
	id := goroutineID()

	st.mu.Lock()
	switch {
	case s.active:
		st.mu.Unlock()
		return ErrScopeActive
	case st.head.Load() != nil && st.owner != id:
		st.mu.Unlock()
		return ErrScopeBusy
	}
	buffer := &scopeBuffer{
		parent: st.head.Load(),
		levels: snapshotLevels(s.core.outer),
	}
	if st.head.Load() == nil {
		st.owner = id
	}
	st.head.Store(buffer)
	s.active = true
	s.buffer = buffer
	st.mu.Unlock()

	finished := false
	defer func() {
		if r := recover(); r != nil {
			// Panic (including a Panic/Fatal terminal hook, or a stubbed
			// os.Exit that panics): discard, then propagate.
			s.exit(false)
			panic(r)
		}
		if !finished {
			s.exit(false)
		}
	}()

	retErr = fn()
	if retErr != nil {
		s.exit(false)
		finished = true
		return retErr
	}

	// A Fatal-level call whose terminal hook did not terminate the process
	// (e.g. a stubbed or overridden hook) must still discard the buffer.
	buffer.mu.Lock()
	terminal := buffer.terminal
	buffer.mu.Unlock()
	s.exit(!terminal)
	finished = true
	return nil
}

// exit pops this scope's buffer. On commit, a nested buffer's entries are
// handed to its parent buffer; a top-level buffer replays them to outer. On
// discard (or for an empty commit) the entries are dropped.
func (s *Scope) exit(commit bool) {
	st := s.core.state
	st.mu.Lock()

	if !s.active || st.head.Load() != s.buffer {
		// Scopes must exit in LIFO order from the goroutine that entered
		// them; nothing else is legal given Enter's guards.
		st.mu.Unlock()
		s.active = false
		s.buffer = nil
		return
	}

	buffer := s.buffer
	parent := buffer.parent

	buffer.mu.Lock()
	entries := buffer.entries
	buffer.entries = nil
	buffer.mu.Unlock()

	st.head.Store(parent)
	if parent == nil {
		st.owner = 0
	}
	s.active = false
	s.buffer = nil

	if commit && parent != nil && len(entries) > 0 {
		parent.mu.Lock()
		parent.entries = append(parent.entries, entries...)
		parent.mu.Unlock()
		entries = nil
	}
	st.mu.Unlock()

	if commit && parent == nil && len(entries) > 0 {
		s.core.replay(entries)
	}
}

// replay feeds committed entries back through the outer Core's Check/Write
// path, in the order they were buffered. Entries the outer Core rejects at
// replay (for example a sampler dropping them) are not written.
func (c *ScopeCore) replay(entries []bufferedEntry) {
	for i := range entries {
		ent := entries[i].ent
		ce := c.outer.Check(ent, nil)
		if ce == nil {
			continue
		}
		ce.Write(entries[i].fields...)
	}
}

// goroutineID returns the calling goroutine's ID. It is only used on the
// scope enter/exit cold path to distinguish nested entry (same goroutine)
// from concurrent entry into a family owned by another goroutine.
func goroutineID() int64 {
	var buf [64]byte
	n := runtime.Stack(buf[:], false)
	// The prefix is "goroutine " followed by the decimal ID.
	b := buf[len("goroutine "):n]
	end := 0
	for end < len(b) && b[end] >= '0' && b[end] <= '9' {
		end++
	}
	id, err := strconv.ParseInt(string(b[:end]), 10, 64)
	if err != nil {
		return -1
	}
	return id
}
