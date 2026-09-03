import { useAccessibility } from '../contexts/AccessibilityContext'
import { useLanguage } from '../contexts/LanguageContext'

export default function AccessibilityControls({ compact = false }) {
  const { fontSize, highContrast, updatePreference } = useAccessibility()
  const { t } = useLanguage()
  if (compact)
    return (
      <button
        type="button"
        className={`contrast-toggle ${highContrast ? 'active' : ''}`}
        aria-pressed={highContrast}
        onClick={() => updatePreference('highContrast', !highContrast)}
      >
        <span aria-hidden="true">◐</span> <span>{t('accessibility.highContrast')}</span>
      </button>
    )
  return (
    <div className="accessibility-controls">
      <fieldset>
        <legend>{t('accessibility.textSize')}</legend>
        <div className="segmented-control">
          {[
            ['normal', 'A', 'accessibility.normal'],
            ['large', 'A', 'accessibility.large'],
            ['xlarge', 'A', 'accessibility.extraLarge'],
          ].map(([value, label, accessibleKey]) => (
            <button
              type="button"
              key={value}
              className={fontSize === value ? 'active' : ''}
              aria-pressed={fontSize === value}
              aria-label={t('accessibility.transcriptText', t(accessibleKey))}
              onClick={() => updatePreference('fontSize', value)}
            >
              <span className={`font-demo font-demo--${value}`}>{label}</span>
              <small>{t(accessibleKey)}</small>
            </button>
          ))}
        </div>
      </fieldset>
      <label className="toggle-row">
        <span>
          <strong>{t('accessibility.highContrast')}</strong>
          <small>{t('accessibility.highContrastDescription')}</small>
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
