import { useState } from 'react'
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
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
  const [open, setOpen] = useState(false)
  const nav = user.role === 'TEACHER' ? teacherNav : studentNav
  const signOut = async () => {
    await logout()
    navigate('/login', { replace: true })
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
        </div>
      </aside>
      <main id="main-content" className="main-content" tabIndex="-1">
        <Outlet />
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
