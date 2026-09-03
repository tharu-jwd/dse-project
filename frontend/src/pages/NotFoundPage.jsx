import { Link } from 'react-router-dom'
import notFoundBackground from '../assets/4.jpg'
import { EmptyState } from '../components/UI'
import { useLanguage } from '../contexts/LanguageContext'

export default function NotFoundPage() {
  const { t } = useLanguage()
  return (
    <main
      className="standalone-page has-bg-image"
      style={{ backgroundImage: `url(${notFoundBackground})` }}
    >
      <EmptyState
        icon="search"
        title={t('notFound.title')}
        message={t('notFound.message')}
        action={
          <Link className="button button--primary" to="/dashboard">
            {t('notFound.goToDashboard')}
          </Link>
        }
      />
    </main>
  )
}
