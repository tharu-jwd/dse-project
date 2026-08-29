import { Link } from 'react-router-dom'
import notFoundBackground from '../assets/4.jpg'
import { EmptyState } from '../components/UI'

export default function NotFoundPage() {
  return (
    <main
      className="standalone-page has-bg-image"
      style={{ backgroundImage: `url(${notFoundBackground})` }}
    >
      <EmptyState
        icon="search"
        title="Page not found"
        message="The page you requested does not exist or may have moved."
        action={
          <Link className="button button--primary" to="/dashboard">
            Go to dashboard
          </Link>
        }
      />
    </main>
  )
}
