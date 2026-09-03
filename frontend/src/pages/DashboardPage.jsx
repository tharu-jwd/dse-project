import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import dashboardBackground from '../assets/1.jpg'
import heroImage from '../assets/astronaut.png'
import Icon from '../components/Icon'
import { Loading, StatusBadge } from '../components/UI'
import { useAuth } from '../contexts/AuthContext'

const actions = {
  STUDENT: [
    [
      'upload',
      'Lecture captioning',
      'Upload a lecture and turn Sinhala speech into clear captions.',
      '/lectures/new',
      'Upload lecture',
    ],
    [
      'quiz',
      'My quizzes',
      'Record spoken answers and review the transcript before submitting.',
      '/quizzes',
      'View quizzes',
    ],
    [
      'mic',
      'Self-study notes',
      'Speak your thoughts and save them as editable study notes.',
      '/notes/new',
      'Create note',
    ],
    [
      'file',
      'Transcript library',
      'Find, review and export your saved transcripts.',
      '/transcripts',
      'Open library',
    ],
  ],
  TEACHER: [
    [
      'upload',
      'Upload lecture',
      'Create accessible Sinhala captions from a recorded lecture.',
      '/lectures/new',
      'Upload lecture',
    ],
    [
      'quiz',
      'Manage quizzes',
      'Create, edit and publish speech-based assessments.',
      '/teacher/quizzes',
      'Manage quizzes',
    ],
    [
      'users',
      'Review submissions',
      'Read answers and leave marks and feedback.',
      '/teacher/submissions',
      'Review work',
    ],
    [
      'file',
      'Transcript library',
      'Find, review and export lecture transcripts.',
      '/transcripts',
      'Open library',
    ],
  ],
}

export default function DashboardPage() {
  const { user } = useAuth()
  const [all, setAll] = useState(null)
  useEffect(() => {
    let active = true
    api
      .getTranscripts()
      .then((items) => active && setAll(items))
      .catch(() => active && setAll([]))
    return () => {
      active = false
    }
  }, [])
  const recent = all?.slice(0, 3) ?? null
  const stats = all && {
    total: all.length,
    finalized: all.filter((item) => item.status === 'FINALIZED').length,
    inProgress: all.filter((item) => item.status !== 'FINALIZED').length,
  }
  const date = new Intl.DateTimeFormat('en-GB', {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
  }).format(new Date())
  const firstAction = actions[user.role][0]
  return (
    <div className="page has-bg-image" style={{ backgroundImage: `url(${dashboardBackground})` }}>
      <section className="hero-banner">
        <span className="eyebrow">{date}</span>
        <h1>Ayubowan, {user.name.split(' ')[0]}!</h1>
        <p>
          {user.role === 'TEACHER'
            ? 'Make today’s learning more accessible for every student.'
            : 'What would you like to work on today?'}
        </p>
        <div className="button-row">
          <Link className="button button--primary" to={firstAction[3]}>
            {firstAction[4]}
          </Link>
          <Link className="button button--secondary" to="/transcripts">
            View library
          </Link>
        </div>
      </section>
      <section className="section">
        <div className="section-heading">
          <h2>{user.role === 'TEACHER' ? 'Quick actions' : 'Mission launchpad'}</h2>
        </div>
        <div className="quick-grid" aria-label="Quick actions">
          {actions[user.role].map(([icon, title, description, to, label], index) => (
            <Link className={`quick-card quick-card--${index + 1}`} to={to} key={to}>
              <span className="quick-card__icon">
                <Icon name={icon} size={26} />
              </span>
              <h2>{title}</h2>
              <p>{description}</p>
              <span className="quick-card__link">
                {label} <Icon name="arrow" size={16} />
              </span>
            </Link>
          ))}
        </div>
      </section>
      <section className="section dashboard-columns">
        <div>
          <div className="section-heading">
            <h2>Your stats</h2>
          </div>
          {stats ? (
            <div className="stat-grid">
              <div className="stat-card">
                <span className="stat-card__icon">
                  <Icon name="file" size={22} />
                </span>
                <div>
                  <p>Total transcripts</p>
                  <p>{stats.total}</p>
                </div>
              </div>
              <div className="stat-card stat-card--2">
                <span className="stat-card__icon">
                  <Icon name="check" size={22} />
                </span>
                <div>
                  <p>Finalized</p>
                  <p>{stats.finalized}</p>
                </div>
              </div>
              <div className="stat-card stat-card--3">
                <span className="stat-card__icon">
                  <Icon name="clock" size={22} />
                </span>
                <div>
                  <p>In progress</p>
                  <p>{stats.inProgress}</p>
                </div>
              </div>
            </div>
          ) : (
            <Loading label="Loading your stats…" />
          )}
        </div>
        <div className="recent-section">
          <div className="section-heading">
            <div>
              <span className="eyebrow">Pick up where you left off</span>
              <h2>Recent transcripts</h2>
            </div>
            <Link className="text-link" to="/transcripts">
              View all <Icon name="arrow" size={15} />
            </Link>
          </div>
          {recent === null ? (
            <Loading label="Loading recent transcripts…" />
          ) : recent.length ? (
            <div className="recent-list">
              {recent.map((item) => (
                <Link to={`/transcripts/${item.id}`} key={item.id} className="recent-item">
                  <span className={`document-icon document-icon--${item.type.toLowerCase()}`}>
                    <Icon
                      name={
                        item.type === 'NOTE'
                          ? 'mic'
                          : item.type === 'QUIZ_ANSWER'
                            ? 'quiz'
                            : 'file'
                      }
                    />
                  </span>
                  <span>
                    <strong>{item.title}</strong>
                    <small>
                      {item.type.replace('_', ' ').toLowerCase()} ·{' '}
                      {new Intl.DateTimeFormat('en-GB', {
                        day: 'numeric',
                        month: 'short',
                        year: 'numeric',
                      }).format(new Date(item.date))}
                    </small>
                  </span>
                  <StatusBadge status={item.status} />
                  <Icon name="more" size={17} />
                </Link>
              ))}
            </div>
          ) : (
            <p className="muted">
              No transcripts yet. Start by uploading a lecture or recording a note.
            </p>
          )}
        </div>
      </section>
      <aside className="tip-card">
        <span aria-hidden="true">සිං</span>
        <div>
          <strong>Quick Start is available in English and Sinhala</strong>
          <p>Learn how to upload a lecture, record an answer and correct low-confidence words.</p>
        </div>
        <Link className="button button--secondary" to="/help">
          View guide
        </Link>
      </aside>
    </div>
  )
}
