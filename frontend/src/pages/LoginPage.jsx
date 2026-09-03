import { useState } from 'react'
import { Navigate, useLocation, useNavigate } from 'react-router-dom'
import { USE_MOCK_API } from '../api'
import loginBackground from '../assets/4.jpg'
import { Logo } from '../components/AppShell'
import Icon from '../components/Icon'
import { Alert } from '../components/UI'
import { useAuth } from '../contexts/AuthContext'
import { useLanguage } from '../contexts/LanguageContext'

export default function LoginPage() {
  const { user, login } = useAuth()
  const { t } = useLanguage()
  const navigate = useNavigate()
  const location = useLocation()
  const [form, setForm] = useState({ email: '', password: '' })
  const [errors, setErrors] = useState({})
  const [serverError, setServerError] = useState('')
  const [loading, setLoading] = useState(false)
  if (user) return <Navigate to="/dashboard" replace />
  const submit = async (event) => {
    event.preventDefault()
    const next = {}
    if (!form.email.trim()) next.email = t('login.emailRequired')
    if (!form.password) next.password = t('login.passwordRequired')
    setErrors(next)
    setServerError('')
    if (Object.keys(next).length) return
    setLoading(true)
    try {
      await login(form.email.trim(), form.password)
      navigate(location.state?.from?.pathname || '/dashboard', { replace: true })
    } catch (error) {
      setServerError(error.code === 'NETWORK_ERROR' ? t('login.networkError') : error.message)
    } finally {
      setLoading(false)
    }
  }
  const fillDemo = (role) => setForm({ email: `${role}@sinhaspeech.lk`, password: 'demo123' })
  return (
    <main className="login-page">
      <section
        className="login-intro has-bg-image"
        style={{ backgroundImage: `url(${loginBackground})` }}
      >
        <Logo />
        <div className="login-intro__content">
          <span className="eyebrow eyebrow--light">{t('login.tagline')}</span>
          <h1>
            {t('login.heroTitle')} <em>{t('login.heroEmphasis')}</em>
          </h1>
          <p>{t('login.heroBody')}</p>
          <ul>
            <li>
              <Icon name="check" /> {t('login.featureLectures')}
            </li>
            <li>
              <Icon name="check" /> {t('login.featureQuiz')}
            </li>
            <li>
              <Icon name="check" /> {t('login.featureNotes')}
            </li>
          </ul>
        </div>
        <p className="login-intro__footer">සිංහලෙන් ඉගෙනීමට සැමට අවස්ථාවක්</p>
      </section>
      <section className="login-panel">
        <div className="login-card">
          <div>
            <span className="eyebrow">{t('login.welcomeBack')}</span>
            <h2>{t('login.signInTitle')}</h2>
            <p>{t('login.signInSubtitle')}</p>
          </div>
          {serverError && <Alert title={t('login.signInFailedTitle')}>{serverError}</Alert>}
          <form onSubmit={submit} noValidate>
            <div className="field">
              <label htmlFor="email">{t('login.emailLabel')}</label>
              <input
                id="email"
                type="email"
                autoComplete="email"
                value={form.email}
                onChange={(e) => setForm({ ...form, email: e.target.value })}
                aria-invalid={Boolean(errors.email)}
                aria-describedby={errors.email ? 'email-error' : undefined}
                placeholder={t('login.emailPlaceholder')}
              />
              {errors.email && (
                <span id="email-error" className="field-error" role="alert">
                  {errors.email}
                </span>
              )}
            </div>
            <div className="field">
              <label htmlFor="password">{t('login.passwordLabel')}</label>
              <input
                id="password"
                type="password"
                autoComplete="current-password"
                value={form.password}
                onChange={(e) => setForm({ ...form, password: e.target.value })}
                aria-invalid={Boolean(errors.password)}
                aria-describedby={errors.password ? 'password-error' : undefined}
                placeholder={t('login.passwordPlaceholder')}
              />
              {errors.password && (
                <span id="password-error" className="field-error" role="alert">
                  {errors.password}
                </span>
              )}
            </div>
            <button
              className="button button--primary button--full"
              type="submit"
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner spinner--small" /> {t('login.signingIn')}
                </>
              ) : (
                <>
                  {t('login.signIn')} <Icon name="arrow" size={18} />
                </>
              )}
            </button>
          </form>
          {USE_MOCK_API && (
            <div className="demo-credentials">
              <div className="section-divider">
                <span>{t('login.demoAccounts')}</span>
              </div>
              <p>{t('login.demoChoose')}</p>
              <div>
                <button type="button" onClick={() => fillDemo('student')}>
                  <span className="demo-avatar demo-avatar--student">ST</span>
                  <span>
                    <strong>{t('login.studentDemo')}</strong>
                    <small>student@sinhaspeech.lk</small>
                  </span>
                  <Icon name="arrow" size={16} />
                </button>
                <button type="button" onClick={() => fillDemo('teacher')}>
                  <span className="demo-avatar demo-avatar--teacher">TE</span>
                  <span>
                    <strong>{t('login.teacherDemo')}</strong>
                    <small>teacher@sinhaspeech.lk</small>
                  </span>
                  <Icon name="arrow" size={16} />
                </button>
              </div>
              <small>
                {t('login.demoPasswordLabel')} <strong>demo123</strong>
              </small>
            </div>
          )}
        </div>
        <p className="login-help">
          <Icon name="help" size={16} /> {t('login.needHelp')}
        </p>
      </section>
    </main>
  )
}
