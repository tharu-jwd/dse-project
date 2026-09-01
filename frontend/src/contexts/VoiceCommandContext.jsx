import { createContext, useContext, useEffect, useMemo, useRef, useState } from 'react'
import useVoiceCommands from '../hooks/useVoiceCommands'
import { useAccessibility } from './AccessibilityContext'

const VoiceCommandContext = createContext(null)

const DEFAULT_HINT = 'Voice commands are on. Nothing to say on this page yet.'

/**
 * One voice-command session for the whole app, not one per page. A
 * student navigating entirely by voice needs the mic (and the "it's
 * listening" indicator) to exist everywhere, not just on the couple of
 * pages that happen to interpret commands - so this is mounted once in
 * AppShell, above the router's <Outlet>, and starts/stops purely off
 * interactionMode. Individual pages don't open their own session; they
 * just register what a recognized command should do here while they're
 * mounted, via useVoiceCommandHandler below.
 */
export function VoiceCommandProvider({ children }) {
  const { interactionMode } = useAccessibility()
  const commandHandlerRef = useRef(null)
  const commandMaybeHandlerRef = useRef(null)
  const [hint, setHint] = useState(DEFAULT_HINT)

  const voice = useVoiceCommands({
    onCommand: (command) => commandHandlerRef.current?.(command),
    onCommandMaybe: (message) => commandMaybeHandlerRef.current?.(message),
  })

  useEffect(() => {
    if (interactionMode === 'command') {
      if (voice.status === 'idle') voice.start()
    } else if (voice.isListening) {
      voice.stop()
    }
  }, [interactionMode]) // eslint-disable-line react-hooks/exhaustive-deps

  const value = useMemo(
    () => ({
      voice,
      hint,
      setHint,
      registerCommandHandler: (fn) => {
        commandHandlerRef.current = fn
      },
      registerCommandMaybeHandler: (fn) => {
        commandMaybeHandlerRef.current = fn
      },
    }),
    [voice, hint],
  )

  return <VoiceCommandContext.Provider value={value}>{children}</VoiceCommandContext.Provider>
}

export function useVoiceCommand() {
  return useContext(VoiceCommandContext)
}

/**
 * A page calls this with its own onCommand logic (a plain inline function,
 * same shape as the old per-page useVoiceCommands({ onCommand }) call) to
 * become the active interpreter for recognized commands while mounted.
 * Pass `null` to explicitly opt out (e.g. TranscriptEditor in compact
 * mode, nested inside a page that already owns the handler) rather than
 * skipping the call - hooks can't be called conditionally.
 */
export function useVoiceCommandHandler(onCommand) {
  const { registerCommandHandler } = useVoiceCommand()
  useEffect(() => {
    // Passing null means "this component doesn't own the handler" (e.g.
    // TranscriptEditor in compact mode, nested inside a page that does) -
    // it must skip touching the shared ref entirely rather than
    // registering null, since this component can mount in a *later*
    // commit than its owning parent (e.g. once a quiz answer finishes
    // transcribing). Registering null then would clobber the parent's
    // already-active handler with nothing to restore it afterwards.
    if (!onCommand) return
    registerCommandHandler(onCommand)
    return () => registerCommandHandler(null)
  }, [onCommand]) // eslint-disable-line react-hooks/exhaustive-deps
}

export function useVoiceCommandMaybeHandler(onCommandMaybe) {
  const { registerCommandMaybeHandler } = useVoiceCommand()
  useEffect(() => {
    if (!onCommandMaybe) return
    registerCommandMaybeHandler(onCommandMaybe)
    return () => registerCommandMaybeHandler(null)
  }, [onCommandMaybe]) // eslint-disable-line react-hooks/exhaustive-deps
}

/** Sets the corner badge's subtitle for as long as this page is mounted,
 * restoring the generic default on unmount. Pass null/falsy to opt out
 * entirely (see useVoiceCommandHandler for why that must be a no-op, not
 * "set the default"). */
export function useVoiceCommandHint(text) {
  const { setHint } = useVoiceCommand()
  useEffect(() => {
    if (!text) return
    setHint(text)
    return () => setHint(DEFAULT_HINT)
  }, [text]) // eslint-disable-line react-hooks/exhaustive-deps
}
