import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import settingsBackground from '../assets/3.jpg'
import AccessibilityControls from '../components/AccessibilityControls'
import Icon from '../components/Icon'
import { PageHeader } from '../components/UI'
import { useAccessibility } from '../contexts/AccessibilityContext'
import { useAuth } from '../contexts/AuthContext'
import { useLanguage } from '../contexts/LanguageContext'
import { useToast } from '../contexts/ToastContext'

const LANGUAGE_LABELS = { si: 'Sinhala', en: 'English' }

export function SettingsPage() {
  const { confidenceThreshold, interactionMode, updatePreference } = useAccessibility()
  const { user } = useAuth()
  const { showToast } = useToast()
  const { language, setLanguage, t } = useLanguage()

  const [commandLanguage, setCommandLanguage] = useState(null)
  const [switchingLanguage, setSwitchingLanguage] = useState(false)

  useEffect(() => {
    if (user.role !== 'STUDENT') return
    api
      .getVoiceEnrollmentStatus()
      .then((data) => setCommandLanguage(data.activeLanguage))
      .catch(() => {})
  }, [user.role])

  const chooseCommandLanguage = async (language) => {
    if (language === commandLanguage) return
    setSwitchingLanguage(true)
    try {
      const result = await api.setActiveCommandLanguage(language)
      setCommandLanguage(result.activeLanguage)
      showToast(`${LANGUAGE_LABELS[language]} is now your voice command language.`)
    } catch (cause) {
      showToast(cause.message || 'Could not switch the command language.', 'error')
    } finally {
      setSwitchingLanguage(false)
    }
  }
  return (
    <div
      className="page page--narrow has-bg-image"
      style={{ backgroundImage: `url(${settingsBackground})` }}
    >
      <PageHeader
        eyebrow={t('settings.eyebrow')}
        title={t('settings.title')}
        description={t('settings.description')}
      />
      <section className="settings-card">
        <h2>{t('settings.languageTitle')}</h2>
        <p className="muted" style={{ fontSize: '0.85rem', marginBottom: 16 }}>
          {t('settings.languageDescription')}
        </p>
        <div className="mode-toggle" role="radiogroup" aria-label={t('settings.languageTitle')}>
          {(['en', 'si']).map((lng) => (
            <button
              key={lng}
              type="button"
              role="radio"
              aria-checked={language === lng}
              className={`mode-toggle__option ${language === lng ? 'active' : ''}`}
              onClick={() => setLanguage(lng)}
            >
              <Icon name={language === lng ? 'check' : 'mic'} size={17} />
              <span>
                <strong>{lng === 'en' ? t('lang.english') : t('lang.sinhala')}</strong>
              </span>
            </button>
          ))}
        </div>
      </section>
      <section className="settings-card">
        <h2>{t('settings.readingPreferences')}</h2>
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
          {interactionMode === 'command' && (
            <div className="voice-setup-options" style={{ marginTop: 16 }}>
              {(['si', 'en']).map((lng) => (
                <button
                  key={lng}
                  type="button"
                  role="radio"
                  aria-checked={commandLanguage === lng}
                  className="voice-setup-option"
                  disabled={switchingLanguage || commandLanguage === null}
                  style={
                    commandLanguage === lng
                      ? { borderColor: 'var(--teal)', background: 'var(--teal-soft)' }
                      : undefined
                  }
                  onClick={() => chooseCommandLanguage(lng)}
                >
                  <span className="voice-setup-option__icon">
                    <Icon name={commandLanguage === lng ? 'check' : 'mic'} size={17} />
                  </span>
                  <span>
                    <strong>{LANGUAGE_LABELS[lng]}</strong>
                    <small>
                      {commandLanguage === null
                        ? 'Loading…'
                        : commandLanguage === lng
                          ? 'Active for your voice commands'
                          : 'Switch to this language'}
                    </small>
                  </span>
                </button>
              ))}
            </div>
          )}
          <p className="muted" style={{ fontSize: '0.85rem', marginTop: 16 }}>
            While taking a self-study note, say a command like "delete" or "stop" to control the
            app hands-free - this always works, in either mode.
            {interactionMode === 'command' &&
              " Haven't recorded your voice yet? Set up samples for a language below."}
          </p>
        </section>
      )}
      {user.role === 'STUDENT' && (
        <section className="settings-card">
          <h2>Voice command setup</h2>
          <p className="muted" style={{ fontSize: '0.85rem', marginBottom: 16 }}>
            Record yourself saying each command a few times. SinhaSpeech turns each recording into
            a voice fingerprint and stores it securely against your account, so it can recognize
            commands by how you say them, not just the words - a second check alongside
            word-matching, useful if your speech is transcribed inconsistently. This is optional:
            voice commands already work from the words alone without enrolling.
          </p>
          <div className="voice-setup-options">
            <Link className="voice-setup-option" to="/settings/voice-commands?language=si">
              <span className="voice-setup-option__icon">
                <Icon name="mic" size={17} />
              </span>
              <span>
                <strong>Sinhala</strong>
                <small>Record, compare and manage your Sinhala command samples.</small>
              </span>
            </Link>
            <Link className="voice-setup-option" to="/settings/voice-commands?language=en">
              <span className="voice-setup-option__icon">
                <Icon name="mic" size={17} />
              </span>
              <span>
                <strong>English</strong>
                <small>Record, compare and manage your English command samples.</small>
              </span>
            </Link>
          </div>
        </section>
      )}
      <button
        type="button"
        className="button button--primary"
        onClick={() => showToast(t('settings.autoSavedToast'))}
      >
        <Icon name="check" size={17} /> {t('settings.confirmPreferences')}
      </button>
    </div>
  )
}

const guides = [
  {
    icon: 'dashboard',
    title: 'Find your way around the dashboard',
    en: [
      'Open Dashboard from the sidebar to see your shortcuts and recent activity.',
      'Use New lecture, New note or New quiz to jump straight into a task.',
      'Recent transcripts and submissions appear below, ordered by last update.',
      'The sidebar stays visible on every page so you can switch tasks at any time.',
    ],
    si: [
      'ප්‍රවේශ මාර්ගයෙන් උපකරණ පුවරුව විවෘත කර ඔබේ කෙටිමං සහ මෑත ක්‍රියාකාරකම් බලන්න.',
      'නව දේශනයක්, නව සටහනක් හෝ නව ප්‍රශ්නාවලියක් ඇරඹීමට ඒ බොත්තම් භාවිතා කරන්න.',
      'මෑත පිටපත් සහ ඉදිරිපත් කිරීම් පහළින් අවසන් යාවත්කාලීන කිරීම අනුව පෙන්වයි.',
      'ඕනෑම පිටුවක සිට කාර්යයක් මාරු කිරීමට ප්‍රවේශ මාර්ගය සැමවිටම දිස්වේ.',
    ],
  },
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
    icon: 'file',
    title: 'Find and edit a transcript',
    en: [
      'Open Transcripts to browse every lecture and note you have access to.',
      'Use the search bar and filters to narrow by title, type or status.',
      'Open a transcript to correct text and review words flagged as low-confidence.',
      'Finalize it when ready, then export as TXT, DOCX or PDF.',
    ],
    si: [
      'ඔබට ප්‍රවේශය ඇති සියලුම දේශන සහ සටහන් බැලීමට පිටපත් පිටුව විවෘත කරන්න.',
      'මාතෘකාව, වර්ගය හෝ තත්ත්වය අනුව පෙරීමට සෙවුම් තීරුව සහ පෙරහන් භාවිතා කරන්න.',
      'පිටපතක් විවෘත කර පෙළ නිවැරදි කරන්න සහ අඩු විශ්වාසනීයත්වයෙන් සලකුණු කළ වචන පරීක්ෂා කරන්න.',
      'සූදානම් වූ පසු අවසන් කර, TXT, DOCX හෝ PDF ලෙස බාගන්න.',
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
  {
    icon: 'quiz',
    title: 'Create and publish a quiz (Teacher)',
    en: [
      'Open Manage quizzes from the teacher menu and choose Create quiz.',
      'Add multiple-choice or spoken questions and mark the required ones.',
      'Save as a draft to keep editing, or Publish to make it visible to students.',
      'Track responses from the same quiz list once students start answering.',
    ],
    si: [
      'ගුරු මෙනුවෙන් ප්‍රශ්නාවලි කළමනාකරණය විවෘත කර ප්‍රශ්නාවලියක් සාදන්න තෝරන්න.',
      'බහුවරණ හෝ කථන ප්‍රශ්න එක් කර අනිවාර්ය ඒවා සලකුණු කරන්න.',
      'තවදුරටත් සංස්කරණය කිරීමට කෙටුම්පතක් ලෙස සුරකින්න, නැතහොත් සිසුන්ට පෙනෙන පරිදි පළ කරන්න.',
      'සිසුන් පිළිතුරු දීම ආරම්භ කළ පසු එම ප්‍රශ්නාවලි ලැයිස්තුවෙන්ම පිළිතුරු නිරීක්ෂණය කරන්න.',
    ],
  },
  {
    icon: 'users',
    title: 'Review and mark submissions (Teacher)',
    en: [
      'Open Submissions to see every student response waiting for review.',
      'Open a submission to read the spoken-answer transcript alongside the question.',
      'Give a mark and written feedback for each answer.',
      'Save your review; students can see their marks and feedback once you do.',
    ],
    si: [
      'සමාලෝචනය සඳහා රැඳී සිටින සියලුම සිසු පිළිතුරු බැලීමට ඉදිරිපත් කිරීම් විවෘත කරන්න.',
      'ප්‍රශ්නය සමඟ කථන පිළිතුරේ පිටපත කියවීමට ඉදිරිපත් කිරීමක් විවෘත කරන්න.',
      'සෑම පිළිතුරකටම ලකුණු සහ ලිඛිත ප්‍රතිපෝෂණ ලබා දෙන්න.',
      'ඔබේ සමාලෝචනය සුරකින්න; සුරැකූ පසු සිසුන්ට ඔවුන්ගේ ලකුණු සහ ප්‍රතිපෝෂණ බැලිය හැක.',
    ],
  },
  {
    icon: 'settings',
    title: 'Personalize accessibility and voice commands',
    en: [
      'Open Settings to choose text size, high contrast and confidence-flag sensitivity.',
      'Students can switch to Command mode to add voice-controlled buttons throughout the app.',
      'Enroll a few voice samples per command language so SinhaSpeech recognizes your voice.',
      'Preferences save automatically and apply the next time you open the app.',
    ],
    si: [
      'අකුරු ප්‍රමාණය, විශාල ප්‍රභේදතාවය සහ විශ්වාසනීයත්ව සලකුණු සංවේදීතාව තෝරා ගැනීමට සැකසුම් විවෘත කරන්න.',
      'යෙදුම පුරාම හඬ-පාලිත බොත්තම් එක් කිරීමට සිසුන්ට විධාන ප්‍රකාරය වෙත මාරු විය හැක.',
      'SinhaSpeech ඔබේ හඬ හඳුනා ගැනීමට එක් එක් විධාන භාෂාව සඳහා හඬ නියැදි කිහිපයක් ලියාපදිංචි කරන්න.',
      'මනාපයන් ස්වයංක්‍රීයව සුරැකෙන අතර ඊළඟ වතාවේ යෙදුම විවෘත කරන විට ක්‍රියාත්මක වේ.',
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
