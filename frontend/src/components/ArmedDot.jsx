import { useEffect, useState } from 'react'

/**
 * The green blinking dot shown while the app is "armed": the wake word
 * ("zimi") was heard and a command is expected. Voice sessions announce it
 * with a 'voice-armed' window event carrying the wake window in seconds
 * (0 means a command used it up), so any number of these can sit on a page
 * and all blink together.
 */
export default function ArmedDot({ className = '' }) {
  const [armed, setArmed] = useState(false)

  useEffect(() => {
    let timer
    const onArmed = (event) => {
      window.clearTimeout(timer)
      const seconds = event.detail?.seconds || 0
      setArmed(seconds > 0)
      if (seconds > 0) timer = window.setTimeout(() => setArmed(false), seconds * 1000)
    }
    window.addEventListener('voice-armed', onArmed)
    return () => {
      window.removeEventListener('voice-armed', onArmed)
      window.clearTimeout(timer)
    }
  }, [])

  if (!armed) return null
  return (
    <span
      className={`armed-dot ${className}`.trim()}
      role="status"
      aria-label="Wake word heard - say a command"
      title="Wake word heard - say a command"
    />
  )
}
