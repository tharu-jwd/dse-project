import { Link } from 'react-router-dom'
import settingsBackground from '../assets/3.jpg'
import AccessibilityControls from '../components/AccessibilityControls'
import Icon from '../components/Icon'
import { PageHeader } from '../components/UI'
import { useAccessibility } from '../contexts/AccessibilityContext'
import { useAuth } from '../contexts/AuthContext'
import { useToast } from '../contexts/ToastContext'

export function SettingsPage() {
  const { confidenceThreshold, interactionMode, updatePreference } = useAccessibility()
  const { user } = useAuth()
  const { showToast } = useToast()
  return (
    <div
      className="page page--narrow has-bg-image"
      style={{ backgroundImage: `url(${settingsBackground})` }}
    >
      <PageHeader
        eyebrow="Make SinhaSpeech work for you"
        title="Accessibility settings"
        description="These preferences are saved on this device and applied across the application."
      />
      <section className="settings-card">
        <h2>Reading preferences</h2>
        <AccessibilityControls />
      </section>
      <section className="settings-card">
        <h2>Transcription confidence</h2>
        <div className="threshold-setting">
          <label htmlFor="global-threshold">
            <span>
              <strong>Low-confidence threshold</strong>
              <small>Words below this confidence score will be marked for review.</small>
            </span>
            <output>{Math.round(confidenceThreshold * 100)}%</output>
          </label>
          <input
            id="global-threshold"
            type="range"
            min="0.5"
            max="0.95"
            step="0.05"
            value={confidenceThreshold}
            onChange={(e) => updatePreference('confidenceThreshold', Number(e.target.value))}
          />
          <div className="threshold-scale">
            <span>Fewer flags</span>
            <span>More flags</span>
          </div>
        </div>
      </section>
      {user.role === 'STUDENT' && (
        <section className="settings-card">
          <h2>Interaction mode</h2>
          <p className="muted" style={{ fontSize: '0.85rem', marginBottom: 16 }}>
            Choose how you want to use SinhaSpeech. Command mode is built for students who find
            typing or using a mouse difficult - it adds voice-controlled buttons (say "save",
            "submit", "next" or "previous") wherever they're available. Normal mode hides those
            extra controls and works entirely through the keyboard and mouse as usual.
          </p>
          <div className="mode-toggle" role="radiogroup" aria-label="Interaction mode">
            <button
              type="button"
              role="radio"
              aria-checked={interactionMode === 'normal'}
              className={`mode-toggle__option ${interactionMode === 'normal' ? 'active' : ''}`}
              onClick={() => updatePreference('interactionMode', 'normal')}
            >
              <Icon name="check" size={17} />
              <span>
                <strong>Normal</strong>
                <small>Keyboard &amp; mouse</small>
              </span>
            </button>
            <button
              type="button"
              role="radio"
              aria-checked={interactionMode === 'command'}
              className={`mode-toggle__option ${interactionMode === 'command' ? 'active' : ''}`}
              onClick={() => updatePreference('interactionMode', 'command')}
            >
              <Icon name="mic" size={17} />
              <span>
                <strong>Command mode</strong>
                <small>Voice-controlled</small>
              </span>
            </button>
          </div>
          <p className="muted" style={{ fontSize: '0.85rem', marginTop: 16, marginBottom: 16 }}>
            While taking a self-study note, say a command like "delete" or "stop" to control the
            app hands-free - this always works, in either mode. Enrolling your voice makes
            recognition more reliable for commands - optional, and normal typing/dictation is
            unaffected either way.
          </p>
          <Link className="button button--secondary" to="/settings/voice-commands">
            <Icon name="mic" size={17} /> Set up voice commands
          </Link>
        </section>
      )}
      <button
        type="button"
        className="button button--primary"
        onClick={() => showToast('Your accessibility settings are saved automatically.')}
      >
        <Icon name="check" size={17} /> Confirm preferences
      </button>
    </div>
  )
}

const guides = [
  {
    icon: 'upload',
    title: 'Caption a lecture',
    en: [
      'Open Lecture captioning from the menu.',
      'Choose an audio or video file and add a title.',
      'Wait for processing, then review uncertain words.',
      'Save, finalize and export your transcript.',
    ],
    si: [
      'මෙනුවෙන් දේශන සිරස්තල පිටුව විවෘත කරන්න.',
      'ශ්‍රව්‍ය හෝ වීඩියෝ ගොනුවක් තෝරා මාතෘකාවක් දෙන්න.',
      'සැක සහිත වචන පරීක්ෂා කර නිවැරදි කරන්න.',
      'පිටපත සුරකින්න, අවසන් කරන්න සහ බාගන්න.',
    ],
  },
  {
    icon: 'mic',
    title: 'Create a voice note',
    en: [
      'Open Self-study notes and enter a title.',
      'Allow microphone access and start recording.',
      'Stop, listen and choose Use recording.',
      'Correct the transcript and save it.',
    ],
    si: [
      'ස්වයං අධ්‍යයන සටහන් විවෘත කර මාතෘකාවක් දෙන්න.',
      'මයික්‍රෆෝනයට අවසර දී පටිගත කිරීම අරඹන්න.',
      'නවත්වා සවන් දී පටිගත කිරීම භාවිතා කරන්න.',
      'පිටපත නිවැරදි කර සුරකින්න.',
    ],
  },
  {
    icon: 'quiz',
    title: 'Answer a spoken quiz',
    en: [
      'Open My quizzes and choose a published quiz.',
      'Record one answer for each required question.',
      'Review and correct the Sinhala transcript.',
      'Confirm only after every answer is complete.',
    ],
    si: [
      'මගේ ප්‍රශ්නාවලි විවෘත කර පළ කළ එකක් තෝරන්න.',
      'සෑම අනිවාර්ය ප්‍රශ්නයකටම පිළිතුරක් පටිගත කරන්න.',
      'සිංහල පිටපත පරීක්ෂා කර නිවැරදි කරන්න.',
      'සියලු පිළිතුරු සම්පූර්ණ වූ පසු තහවුරු කරන්න.',
    ],
  },
]

export function HelpPage() {
  return (
    <div className="page has-bg-image" style={{ backgroundImage: `url(${settingsBackground})` }}>
      <PageHeader
        eyebrow="English · සිංහල"
        title="Quick start guide"
        description="Simple instructions for the main SinhaSpeech workflows."
      />
      <div className="help-grid">
        {guides.map((guide) => (
          <article className="help-card" key={guide.title}>
            <span className="help-card__icon">
              <Icon name={guide.icon} />
            </span>
            <h2>{guide.title}</h2>
            <div className="bilingual">
              <section lang="en">
                <span>English</span>
                <ol>
                  {guide.en.map((step) => (
                    <li key={step}>{step}</li>
                  ))}
                </ol>
              </section>
              <section lang="si">
                <span>සිංහල</span>
                <ol>
                  {guide.si.map((step) => (
                    <li key={step}>{step}</li>
                  ))}
                </ol>
              </section>
            </div>
          </article>
        ))}
      </div>
      <section className="support-card">
        <div>
          <Icon name="help" size={24} />
          <span>
            <strong>Still need help?</strong>
            <p>Ask your course administrator or accessibility support contact.</p>
          </span>
        </div>
        <a className="button button--secondary" href="mailto:support@sinhaspeech.lk">
          Email support
        </a>
      </section>
    </div>
  )
}
