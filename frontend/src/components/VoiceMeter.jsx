import { VOICE_METER_BAR_COUNT } from '../hooks/useVoiceCommands'

/**
 * The animated bar meter shown while a voice-command session is listening.
 * `registerBar`/`active` come straight from useVoiceCommands - this is
 * purely presentational so it can be reused on any page with a mic toggle.
 */
export default function VoiceMeter({ registerBar, active, compact = false }) {
  return (
    <div className={`voice-meter ${active ? 'voice-meter--active' : ''} ${compact ? 'voice-meter--compact' : ''}`}>
      {Array.from({ length: VOICE_METER_BAR_COUNT }).map((_, index) => (
        <i key={index} ref={registerBar(index)} />
      ))}
    </div>
  )
}
