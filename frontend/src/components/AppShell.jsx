import { useState } from 'react'
import { NavLink, Outlet, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import AccessibilityControls from './AccessibilityControls'
import Icon from './Icon'

const studentNav = [
  ['dashboard', 'Dashboard', '/dashboard'],
  ['upload', 'Lecture captioning', '/lectures/new'],
  ['quiz', 'My quizzes', '/quizzes'],
  ['mic', 'Self-study notes', '/notes/new'],
  ['file', 'Transcript library', '/transcripts'],
]
const teacherNav = [
  ['dashboard', 'Dashboard', '/dashboard'],
  ['upload', 'Upload lecture', '/lectures/new'],
  ['quiz', 'Manage quizzes', '/teacher/quizzes'],
  ['users', 'Review submissions', '/teacher/submissions'],
  ['file', 'Transcript library', '/transcripts'],
]

export default function AppShell() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const nav = user.role === 'TEACHER' ? teacherNav : studentNav
  const signOut = async () => {
    await logout()
    navigate('/login', { replace: true })
  }
  const activeLabel =
    nav.find(([, , to]) => location.pathname.startsWith(to))?.[1] ||
    (location.pathname.startsWith('/settings')
      ? 'Settings'
      : location.pathname.startsWith('/help')
        ? 'Quick start & help'
        : location.pathname.startsWith('/teacher/submissions')
          ? 'Review submissions'
          : location.pathname.startsWith('/notes')
            ? 'Self-study notes'
            : 'SinhaSpeech')
  const submitSearch = (event) => {
    event.preventDefault()
    if (search.trim()) navigate(`/transcripts?q=${encodeURIComponent(search.trim())}`)
  }
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>
      <header className="mobile-header">
        <button className="icon-button" aria-label="Open navigation" onClick={() => setOpen(true)}>
          <Icon name="menu" />
        </button>
        <Logo />
      </header>
      {open && (
        <button
          className="nav-overlay"
          aria-label="Close navigation"
          onClick={() => setOpen(false)}
        />
      )}
      <aside className={`sidebar ${open ? 'sidebar--open' : ''}`}>
        <div className="sidebar__brand">
          <Logo />
          <button
            className="icon-button sidebar__close"
            aria-label="Close navigation"
            onClick={() => setOpen(false)}
          >
            <Icon name="close" />
          </button>
        </div>
        <nav aria-label="Main navigation">
          {nav.map(([icon, label, to]) => (
            <NavLink key={to} to={to} onClick={() => setOpen(false)}>
              <Icon name={icon} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar__bottom">
          <nav aria-label="Support navigation">
            <NavLink to="/settings">
              <Icon name="settings" />
              <span>Settings</span>
            </NavLink>
            <NavLink to="/help">
              <Icon name="help" />
              <span>Quick start & help</span>
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
              <small>{user.role === 'TEACHER' ? 'Teacher' : 'Student'}</small>
            </div>
            <button
              className="icon-button"
              aria-label="Sign out"
              title="Sign out"
              onClick={signOut}
            >
              <Icon name="logout" />
            </button>
          </div>
          <NavLink className="new-adventure" to="/lectures/new" onClick={() => setOpen(false)}>
            <Icon name="rocket" size={18} />
            <span>New adventure</span>
          </NavLink>
        </div>
      </aside>
      <main id="main-content" className="main-content" tabIndex="-1">
        <header className="topbar">
          <h2>{activeLabel}</h2>
          <form className="topbar__search" role="search" onSubmit={submitSearch}>
            <Icon name="search" size={18} />
            <label className="sr-only" htmlFor="topbar-search">
              Search your transcripts
            </label>
            <input
              id="topbar-search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search your transcripts…"
              type="search"
            />
          </form>
          <div className="topbar__actions">
            <button className="icon-button topbar__bell" aria-label="Notifications" type="button">
              <Icon name="bell" size={19} />
            </button>
            <NavLink className="avatar avatar--header" to="/settings" title="Account settings">
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
              Accessibility statement
            </a>
            <a href="#" onClick={(e) => e.preventDefault()}>
              Privacy policy
            </a>
            <a href="#" onClick={(e) => e.preventDefault()}>
              Terms of service
            </a>
            <NavLink to="/help">Help center</NavLink>
          </div>
          <p>© {new Date().getFullYear()} SinhaSpeech Accessibility.</p>
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
