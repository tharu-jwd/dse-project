import { useState } from 'react'
import { Navigate, useLocation, useNavigate } from 'react-router-dom'
import { USE_MOCK_API } from '../api'
import loginBackground from '../assets/general-background.jpg'
import { Logo } from '../components/AppShell'
import Icon from '../components/Icon'
import { Alert } from '../components/UI'
import { useAuth } from '../contexts/AuthContext'

export default function LoginPage() {
  const { user, login } = useAuth()
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
    if (!form.email.trim()) next.email = 'Email is required.'
    if (!form.password) next.password = 'Password is required.'
    setErrors(next)
    setServerError('')
    if (Object.keys(next).length) return
    setLoading(true)
    try {
      await login(form.email.trim(), form.password)
      navigate(location.state?.from?.pathname || '/dashboard', { replace: true })
    } catch (error) {
      setServerError(
        error.code === 'NETWORK_ERROR'
          ? 'We could not connect to SinhaSpeech. Check your network and try again.'
          : error.message,
      )
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
          <span className="eyebrow eyebrow--light">Sinhala speech accessibility</span>
          <h1>
            Every voice deserves to be <em>understood.</em>
          </h1>
          <p>
            Turn Sinhala speech into clear, editable text for lectures, learning and assessment.
          </p>
          <ul>
            <li>
              <Icon name="check" /> Accessible lecture captions
            </li>
            <li>
              <Icon name="check" /> Spoken quiz answers
            </li>
            <li>
              <Icon name="check" /> Voice-powered study notes
            </li>
          </ul>
        </div>
        <p className="login-intro__footer">සිංහලෙන් ඉගෙනීමට සැමට අවස්ථාවක්</p>
      </section>
      <section className="login-panel">
        <div className="login-card">
          <div>
            <span className="eyebrow">Welcome back</span>
            <h2>Sign in to SinhaSpeech</h2>
            <p>Use your university account to continue.</p>
          </div>
          {serverError && <Alert title="Sign-in unsuccessful">{serverError}</Alert>}
          <form onSubmit={submit} noValidate>
            <div className="field">
              <label htmlFor="email">Email address</label>
              <input
                id="email"
                type="email"
                autoComplete="email"
                value={form.email}
                onChange={(e) => setForm({ ...form, email: e.target.value })}
                aria-invalid={Boolean(errors.email)}
                aria-describedby={errors.email ? 'email-error' : undefined}
                placeholder="name@university.lk"
              />
              {errors.email && (
                <span id="email-error" className="field-error" role="alert">
                  {errors.email}
                </span>
              )}
            </div>
            <div className="field">
              <label htmlFor="password">Password</label>
              <input
                id="password"
                type="password"
                autoComplete="current-password"
                value={form.password}
                onChange={(e) => setForm({ ...form, password: e.target.value })}
                aria-invalid={Boolean(errors.password)}
                aria-describedby={errors.password ? 'password-error' : undefined}
                placeholder="Enter your password"
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
                  <span className="spinner spinner--small" /> Signing in…
                </>
              ) : (
                <>
                  Sign in <Icon name="arrow" size={18} />
                </>
              )}
            </button>
          </form>
          {USE_MOCK_API && (
            <div className="demo-credentials">
              <div className="section-divider">
                <span>Demo accounts</span>
              </div>
              <p>Choose an account to fill in the demo credentials.</p>
              <div>
                <button type="button" onClick={() => fillDemo('student')}>
                  <span className="demo-avatar demo-avatar--student">ST</span>
                  <span>
                    <strong>Student demo</strong>
                    <small>student@sinhaspeech.lk</small>
                  </span>
                  <Icon name="arrow" size={16} />
                </button>
                <button type="button" onClick={() => fillDemo('teacher')}>
                  <span className="demo-avatar demo-avatar--teacher">TE</span>
                  <span>
                    <strong>Teacher demo</strong>
                    <small>teacher@sinhaspeech.lk</small>
                  </span>
                  <Icon name="arrow" size={16} />
                </button>
              </div>
              <small>
                Password for both accounts: <strong>demo123</strong>
              </small>
            </div>
          )}
        </div>
        <p className="login-help">
          <Icon name="help" size={16} /> Need help? Contact your course administrator.
        </p>
      </section>
    </main>
  )
}
