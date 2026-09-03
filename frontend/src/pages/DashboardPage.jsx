import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import dashboardBackground from '../assets/7.jpg'
import heroImage from '../assets/astronaut.png'
import Icon from '../components/Icon'
import { Loading, StatusBadge } from '../components/UI'
import { useAuth } from '../contexts/AuthContext'
import { useLanguage } from '../contexts/LanguageContext'

const actions = {
  STUDENT: [
    [
      'upload',
      'dashboard.student.lectureTitle',
      'dashboard.student.lectureDescription',
      '/lectures/new',
      'dashboard.student.lectureCta',
    ],
    [
      'quiz',
      'dashboard.student.quizTitle',
      'dashboard.student.quizDescription',
      '/quizzes',
      'dashboard.student.quizCta',
    ],
    [
      'mic',
      'dashboard.student.notesTitle',
      'dashboard.student.notesDescription',
      '/notes/new',
      'dashboard.student.notesCta',
    ],
    [
      'file',
      'dashboard.libraryTitle',
      'dashboard.student.libraryDescription',
      '/transcripts',
      'dashboard.libraryCta',
    ],
  ],
  TEACHER: [
    [
      'upload',
      'nav.uploadLecture',
      'dashboard.teacher.lectureDescription',
      '/lectures/new',
      'nav.uploadLecture',
    ],
    [
      'quiz',
      'dashboard.teacher.quizTitle',
      'dashboard.teacher.quizDescription',
      '/teacher/quizzes',
      'dashboard.teacher.quizCta',
    ],
    [
      'users',
      'dashboard.teacher.submissionsTitle',
      'dashboard.teacher.submissionsDescription',
      '/teacher/submissions',
      'dashboard.teacher.submissionsCta',
    ],
    [
      'file',
      'dashboard.libraryTitle',
      'dashboard.teacher.libraryDescription',
      '/transcripts',
      'dashboard.libraryCta',
    ],
  ],
}

export default function DashboardPage() {
  const { user } = useAuth()
  const { t, language } = useLanguage()
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
  const date = new Intl.DateTimeFormat(language === 'si' ? 'si-LK' : 'en-GB', {
    weekday: 'long',
    day: 'numeric',
    month: 'long',
  }).format(new Date())
  const firstAction = actions[user.role][0]
  return (
    <div className="page has-bg-image" style={{ backgroundImage: `url(${dashboardBackground})` }}>
      <section className="hero-banner">
        <span className="eyebrow">{date}</span>
        <h1>{t('dashboard.greeting', user.name.split(' ')[0])}</h1>
        <p>
          {user.role === 'TEACHER' ? t('dashboard.teacherSubtitle') : t('dashboard.studentSubtitle')}
        </p>
        <div className="button-row">
          <Link className="button button--primary" to={firstAction[3]}>
            {t(firstAction[4])}
          </Link>
          <Link className="button button--secondary" to="/transcripts">
            {t('dashboard.viewLibrary')}
          </Link>
        </div>
      </section>
      <section className="section">
        <div className="section-heading">
          <h2>{user.role === 'TEACHER' ? t('dashboard.quickActions') : t('dashboard.missionLaunchpad')}</h2>
        </div>
        <div className="quick-grid" aria-label={t('dashboard.quickActions')}>
          {actions[user.role].map(([icon, titleKey, descriptionKey, to, labelKey], index) => (
            <Link className={`quick-card quick-card--${index + 1}`} to={to} key={to}>
              <span className="quick-card__icon">
                <Icon name={icon} size={26} />
              </span>
              <h2>{t(titleKey)}</h2>
              <p>{t(descriptionKey)}</p>
              <span className="quick-card__link">
                {t(labelKey)} <Icon name="arrow" size={16} />
              </span>
            </Link>
          ))}
        </div>
      </section>
      <section className="section dashboard-columns">
        <div>
          <div className="section-heading">
            <h2>{t('dashboard.yourStats')}</h2>
          </div>
          {stats ? (
            <div className="stat-grid">
              <div className="stat-card">
                <span className="stat-card__icon">
                  <Icon name="file" size={22} />
                </span>
                <div>
                  <p>{t('dashboard.totalTranscripts')}</p>
                  <p>{stats.total}</p>
                </div>
              </div>
              <div className="stat-card stat-card--2">
                <span className="stat-card__icon">
                  <Icon name="check" size={22} />
                </span>
                <div>
                  <p>{t('dashboard.finalized')}</p>
                  <p>{stats.finalized}</p>
                </div>
              </div>
              <div className="stat-card stat-card--3">
                <span className="stat-card__icon">
                  <Icon name="clock" size={22} />
                </span>
                <div>
                  <p>{t('dashboard.inProgress')}</p>
                  <p>{stats.inProgress}</p>
                </div>
              </div>
            </div>
          ) : (
            <Loading label={t('dashboard.loadingStats')} />
          )}
        </div>
        <div className="recent-section">
          <div className="section-heading">
            <div>
              <span className="eyebrow">{t('dashboard.pickUpWhereLeftOff')}</span>
              <h2>{t('dashboard.recentTranscripts')}</h2>
            </div>
            <Link className="text-link" to="/transcripts">
              {t('dashboard.viewAll')} <Icon name="arrow" size={15} />
            </Link>
          </div>
          {recent === null ? (
            <Loading label={t('dashboard.loadingRecent')} />
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
                      {new Intl.DateTimeFormat(language === 'si' ? 'si-LK' : 'en-GB', {
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
            <p className="muted">{t('dashboard.noTranscripts')}</p>
          )}
        </div>
      </section>
      <aside className="tip-card">
        <span aria-hidden="true">සිං</span>
        <div>
          <strong>{t('dashboard.tipTitle')}</strong>
          <p>{t('dashboard.tipBody')}</p>
        </div>
        <Link className="button button--secondary" to="/help">
          {t('dashboard.viewGuide')}
        </Link>
      </aside>
    </div>
  )
}
