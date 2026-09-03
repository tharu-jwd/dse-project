import { useState } from 'react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { useLanguage } from '../contexts/LanguageContext'
import AccessibilityControls from './AccessibilityControls'
import Icon from './Icon'

const studentNav = [
  ['dashboard', 'nav.dashboard', '/dashboard'],
  ['upload', 'nav.lectureCaptioning', '/lectures/new'],
  ['quiz', 'nav.myQuizzes', '/quizzes'],
  ['mic', 'nav.selfStudyNotes', '/notes/new'],
  ['file', 'nav.transcriptLibrary', '/transcripts'],
]
const teacherNav = [
  ['dashboard', 'nav.dashboard', '/dashboard'],
  ['upload', 'nav.uploadLecture', '/lectures/new'],
  ['quiz', 'nav.manageQuizzes', '/teacher/quizzes'],
  ['users', 'nav.reviewSubmissions', '/teacher/submissions'],
  ['file', 'nav.transcriptLibrary', '/transcripts'],
]

export default function AppShell() {
  const { user, logout } = useAuth()
  const { t } = useLanguage()
  const navigate = useNavigate()
  const location = useLocation()
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const nav = user.role === 'TEACHER' ? teacherNav : studentNav
  const signOut = async () => {
    await logout()
    navigate('/login', { replace: true })
  }
  const activeLabelKey =
    nav.find(([, , to]) => location.pathname.startsWith(to))?.[1] ||
    (location.pathname.startsWith('/settings')
      ? 'nav.settings'
      : location.pathname.startsWith('/help')
        ? 'nav.quickStartHelp'
        : location.pathname.startsWith('/teacher/submissions')
          ? 'nav.reviewSubmissions'
          : location.pathname.startsWith('/notes')
            ? 'nav.selfStudyNotes'
            : null)
  const activeLabel = activeLabelKey ? t(activeLabelKey) : 'SinhaSpeech'
  const submitSearch = (event) => {
    event.preventDefault()
    if (search.trim()) navigate(`/transcripts?q=${encodeURIComponent(search.trim())}`)
  }
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        {t('nav.skipToContent')}
      </a>
      <header className="mobile-header">
        <button
          className="icon-button"
          aria-label={t('nav.openNavigation')}
          onClick={() => setOpen(true)}
        >
          <Icon name="menu" />
        </button>
        <Logo />
      </header>
      {open && (
        <button
          className="nav-overlay"
          aria-label={t('nav.closeNavigation')}
          onClick={() => setOpen(false)}
        />
      )}
      <aside className={`sidebar ${open ? 'sidebar--open' : ''}`}>
        <div className="sidebar__brand">
          <Logo />
          <button
            className="icon-button sidebar__close"
            aria-label={t('nav.closeNavigation')}
            onClick={() => setOpen(false)}
          >
            <Icon name="close" />
          </button>
        </div>
        <nav aria-label="Main navigation">
          {nav.map(([icon, labelKey, to]) => (
            <NavLink key={to} to={to} onClick={() => setOpen(false)}>
              <Icon name={icon} />
              <span>{t(labelKey)}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar__bottom">
          <nav aria-label="Support navigation">
            <NavLink to="/settings">
              <Icon name="settings" />
              <span>{t('nav.settings')}</span>
            </NavLink>
            <NavLink to="/help">
              <Icon name="help" />
              <span>{t('nav.quickStartHelp')}</span>
            </NavLink>
          </nav>
          <AccessibilityControls compact />
          <div className="user-card">
            <div className="avatar">
              {user.name
                .split(' ')
                .map((part) => part[0])
                .slice(0, 2)
                .join('')}
            </div>
            <div>
              <strong>{user.name}</strong>
              <small>{user.role === 'TEACHER' ? t('nav.teacher') : t('nav.student')}</small>
            </div>
            <button
              className="icon-button"
              aria-label={t('nav.signOut')}
              title={t('nav.signOut')}
              onClick={signOut}
            >
              <Icon name="logout" />
            </button>
          </div>
          <NavLink className="new-adventure" to="/lectures/new" onClick={() => setOpen(false)}>
            <Icon name="rocket" size={18} />
            <span>{t('nav.newAdventure')}</span>
          </NavLink>
        </div>
      </aside>
      <main id="main-content" className="main-content" tabIndex="-1">
        <header className="topbar">
          <h2>{activeLabel}</h2>
          <form className="topbar__search" role="search" onSubmit={submitSearch}>
            <Icon name="search" size={18} />
            <label className="sr-only" htmlFor="topbar-search">
              {t('nav.searchPlaceholder')}
            </label>
            <input
              id="topbar-search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={t('nav.searchPlaceholder')}
              type="search"
            />
          </form>
          <div className="topbar__actions">
            <button
              className="icon-button topbar__bell"
              aria-label={t('nav.notifications')}
              type="button"
            >
              <Icon name="bell" size={19} />
            </button>
            <NavLink
              className="avatar avatar--header"
              to="/settings"
              title={t('nav.accountSettings')}
            >
              {user.name
                .split(' ')
                .map((part) => part[0])
                .slice(0, 2)
                .join('')}
            </NavLink>
          </div>
        </header>
        <Outlet />
        <footer className="app-footer">
          <div className="app-footer__links">
            <a href="#" onClick={(e) => e.preventDefault()}>
              {t('footer.accessibilityStatement')}
            </a>
            <a href="#" onClick={(e) => e.preventDefault()}>
              {t('footer.privacyPolicy')}
            </a>
            <a href="#" onClick={(e) => e.preventDefault()}>
              {t('footer.termsOfService')}
            </a>
            <NavLink to="/help">{t('footer.helpCenter')}</NavLink>
          </div>
          <p>{t('footer.copyright', new Date().getFullYear())}</p>
        </footer>
      </main>
    </div>
  )
}

export function Logo() {
  return (
    <div className="logo">
      <span className="logo__mark" aria-hidden="true">
        <span />
        <span />
        <span />
        <span />
      </span>
      <span>
        <strong>Sinha</strong>Speech<small>සිංහල කථන සහායක</small>
      </span>
    </div>
  )
}
