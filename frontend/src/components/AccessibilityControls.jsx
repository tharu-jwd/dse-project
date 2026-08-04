import { useAccessibility } from '../contexts/AccessibilityContext'

export default function AccessibilityControls({ compact = false }) {
  const { fontSize, highContrast, updatePreference } = useAccessibility()
  if (compact)
    return (
      <button
        type="button"
        className={`contrast-toggle ${highContrast ? 'active' : ''}`}
        aria-pressed={highContrast}
        onClick={() => updatePreference('highContrast', !highContrast)}
      >
        <span aria-hidden="true">◐</span> <span>High contrast</span>
      </button>
    )
  return (
    <div className="accessibility-controls">
      <fieldset>
        <legend>Transcript text size</legend>
        <div className="segmented-control">
          {[
            ['normal', 'A', 'Normal'],
            ['large', 'A', 'Large'],
            ['xlarge', 'A', 'Extra large'],
          ].map(([value, label, accessible]) => (
            <button
              type="button"
              key={value}
              className={fontSize === value ? 'active' : ''}
              aria-pressed={fontSize === value}
              aria-label={`${accessible} transcript text`}
              onClick={() => updatePreference('fontSize', value)}
            >
              <span className={`font-demo font-demo--${value}`}>{label}</span>
              <small>{accessible}</small>
            </button>
          ))}
        </div>
      </fieldset>
      <label className="toggle-row">
        <span>
          <strong>High contrast</strong>
          <small>Increase contrast throughout the application.</small>
        </span>
        <input
          type="checkbox"
          checked={highContrast}
          onChange={(e) => updatePreference('highContrast', e.target.checked)}
        />
        <span className="toggle" aria-hidden="true" />
      </label>
    </div>
  )
}
