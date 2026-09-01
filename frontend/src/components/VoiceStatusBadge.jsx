import Icon from './Icon'
import VoiceMeter from './VoiceMeter'

/**
 * Fixed-corner "the mic is on" indicator for Command Mode. Listening now
 * starts automatically (see useVoiceCommands auto-start in TranscriptEditor
 * / StudentQuizPages) rather than from a click, so a student has no button
 * press to confirm it's working - this has to be visible regardless of
 * scroll position, not just an inline badge in a toolbar that can scroll
 * out of view. Also doubles as the manual stop/restart control.
 */
export default function VoiceStatusBadge({ voice, hint }) {
  if (voice.status === 'idle') return null

  const label =
    voice.status === 'connecting'
      ? 'Connecting…'
      : voice.status === 'listening'
        ? 'Listening for commands'
        : voice.status === 'stopping'
          ? 'Stopping…'
          : 'Voice commands paused'

  return (
    <div
      className={`voice-status-badge voice-status-badge--${voice.status}`}
      role="status"
      aria-live="polite"
    >
      <span className="voice-status-badge__icon" aria-hidden="true">
        <Icon name={voice.status === 'error' ? 'alert' : 'mic'} size={16} />
      </span>
      <span className="voice-status-badge__text">
        <strong>{label}</strong>
        <small>{voice.status === 'error' ? voice.error || 'Tap to retry.' : hint}</small>
      </span>
      {voice.isListening && (
        <VoiceMeter registerBar={voice.registerBar} active={voice.voiceDetected} compact />
      )}
      <button
        type="button"
        className="voice-status-badge__toggle"
        onClick={voice.isListening ? voice.stop : voice.start}
        disabled={voice.status === 'connecting' || voice.status === 'stopping'}
        aria-label={voice.isListening ? 'Pause voice commands' : 'Restart voice commands'}
      >
        <Icon name={voice.isListening ? 'stop' : 'mic'} size={15} />
      </button>
    </div>
  )
}
