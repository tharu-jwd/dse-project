import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { api, downloadBlob } from '../api'
import editorBackground from '../assets/1.jpg'
import libraryBackground from '../assets/5.png'
import Icon from '../components/Icon'
import TranscriptEditor from '../components/TranscriptEditor'
import { Alert, ConfirmDialog, EmptyState, Loading, PageHeader, StatusBadge } from '../components/UI'
import { useLanguage } from '../contexts/LanguageContext'
import { useToast } from '../contexts/ToastContext'

export function TranscriptLibraryPage() {
  const { t, language } = useLanguage()
  const typeLabel = { LECTURE: t('type.lecture'), NOTE: t('type.studyNote'), QUIZ_ANSWER: t('type.quizAnswer') }
  const [searchParams] = useSearchParams()
  const [items, setItems] = useState(null)
  const [error, setError] = useState('')
  const [search, setSearch] = useState(searchParams.get('q') || '')
  const [type, setType] = useState('ALL')
  const [status, setStatus] = useState('ALL')
  const [deleteTarget, setDeleteTarget] = useState(null)
  const [isDeleting, setIsDeleting] = useState(false)
  const { showToast } = useToast()
  const load = () => {
    setItems(null)
    setError('')
    api
      .getTranscripts()
      .then(setItems)
      .catch((cause) => setError(cause.message))
  }
  useEffect(load, [])
  const filtered = useMemo(
    () =>
      (items || []).filter(
        (item) =>
          item.title.toLowerCase().includes(search.toLowerCase()) &&
          (type === 'ALL' || item.type === type) &&
          (status === 'ALL' || item.status === status),
      ),
    [items, search, type, status],
  )
  const exportItem = async (item) => {
    try {
      const blob = await api.exportTranscript(item.id, 'txt')
      downloadBlob(blob, `${item.title}.txt`)
      showToast(t('library.transcriptDownloaded'))
    } catch (cause) {
      setError(cause.message)
    }
  }
  const deleteItem = async () => {
    if (isDeleting) return
    const target = deleteTarget
    setIsDeleting(true)
    try {
      try {
        await api.deleteTranscript(target.id)
      } catch (cause) {
        if (!/not found/i.test(cause.message)) throw cause
      }
      setItems((prev) => (prev || []).filter((entry) => entry.id !== target.id))
      showToast(t('library.transcriptDeleted'))
      setDeleteTarget(null)
    } catch (cause) {
      setError(cause.message)
    } finally {
      setIsDeleting(false)
    }
  }
  return (
    <div className="page has-bg-image" style={{ backgroundImage: `url(${libraryBackground})` }}>
      <PageHeader
        eyebrow={t('library.eyebrow')}
        title={t('library.title')}
        description={t('library.description')}
      />
      <section className="library-tools" aria-label={t('library.filtersLabel')}>
        <label className="search-field">
          <span className="sr-only">{t('library.searchTranscripts')}</span>
          <Icon name="search" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder={t('library.searchPlaceholder')}
          />
        </label>
        <label>
          <span className="sr-only">{t('library.filterByType')}</span>
          <select value={type} onChange={(e) => setType(e.target.value)}>
            <option value="ALL">{t('library.allTypes')}</option>
            <option value="LECTURE">{t('library.lectures')}</option>
            <option value="NOTE">{t('library.studyNotes')}</option>
            <option value="QUIZ_ANSWER">{t('library.quizAnswers')}</option>
          </select>
        </label>
        <label>
          <span className="sr-only">{t('library.filterByStatus')}</span>
          <select value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="ALL">{t('library.allStatuses')}</option>
            <option value="DRAFT">{t('status.draft')}</option>
            <option value="FINALIZED">{t('status.finalized')}</option>
            <option value="PROCESSING">{t('status.processing')}</option>
            <option value="FAILED">{t('status.failed')}</option>
          </select>
        </label>
      </section>
      {error && (
        <Alert title={t('library.couldNotLoad')}>
          {error}
          <div>
            <button className="button button--secondary button--small" onClick={load}>
              {t('library.tryAgain')}
            </button>
          </div>
        </Alert>
      )}
      {items === null && !error ? (
        <Loading label={t('library.loading')} />
      ) : filtered.length ? (
        <div className="table-card">
          <table>
            <thead>
              <tr>
                <th>{t('library.colTranscript')}</th>
                <th>{t('library.colType')}</th>
                <th>{t('library.colDateCreated')}</th>
                <th>{t('library.colStatus')}</th>
                <th>
                  <span className="sr-only">{t('library.colActions')}</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((item) => (
                <tr key={item.id}>
                  <td>
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
                    <strong>{item.title}</strong>
                  </td>
                  <td>{typeLabel[item.type]}</td>
                  <td>
                    {new Intl.DateTimeFormat(language === 'si' ? 'si-LK' : 'en-GB', {
                      day: 'numeric',
                      month: 'short',
                      year: 'numeric',
                    }).format(new Date(item.date))}
                  </td>
                  <td>
                    <StatusBadge status={item.status} />
                  </td>
                  <td>
                    <div className="row-actions">
                      <Link
                        className="button button--secondary button--small"
                        to={`/transcripts/${item.id}`}
                      >
                        {t('library.open')}
                      </Link>
                      {item.status === 'FINALIZED' && (
                        <button
                          className="icon-button"
                          title={t('library.exportTxt')}
                          aria-label={t('library.exportLabel', item.title)}
                          onClick={() => exportItem(item)}
                        >
                          <Icon name="download" size={18} />
                        </button>
                      )}
                      <button
                        className="icon-button"
                        title={t('library.delete')}
                        aria-label={t('library.deleteLabel', item.title)}
                        onClick={() => setDeleteTarget(item)}
                      >
                        <Icon name="trash" size={18} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <EmptyState
          icon="search"
          title={items?.length ? t('library.noMatchingTranscripts') : t('library.libraryReady')}
          message={
            items?.length ? t('library.tryChangingFilters') : t('library.emptyLibraryMessage')
          }
          action={
            !items?.length && (
              <Link className="button button--primary" to="/lectures/new">
                {t('library.createTranscript')}
              </Link>
            )
          }
        />
      )}
      <ConfirmDialog
        open={Boolean(deleteTarget)}
        title={t('library.deleteTranscriptTitle')}
        message={t('library.deleteConfirmMessage', deleteTarget?.title)}
        confirmLabel={t('library.deleteTranscriptConfirm')}
        dangerous
        busy={isDeleting}
        onConfirm={deleteItem}
        onCancel={() => setDeleteTarget(null)}
      />
    </div>
  )
}

export function TranscriptPage() {
  const { t } = useLanguage()
  const typeLabel = { LECTURE: t('type.lecture'), NOTE: t('type.studyNote'), QUIZ_ANSWER: t('type.quizAnswer') }
  const { id } = useParams()
  const navigate = useNavigate()
  const [item, setItem] = useState(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let active = true
    api
      .getTranscript(id)
      .then((value) => active && setItem(value))
      .catch((cause) => active && setError(cause.message))
    return () => {
      active = false
    }
  }, [id])
  if (error)
    return (
      <div
        className="page page--narrow has-bg-image"
        style={{ backgroundImage: `url(${editorBackground})` }}
      >
        <Alert title={t('library.couldNotOpen')}>{error}</Alert>
        <button className="button button--secondary" onClick={() => navigate('/transcripts')}>
          {t('library.backToLibrary')}
        </button>
      </div>
    )
  if (!item)
    return (
      <div className="page has-bg-image" style={{ backgroundImage: `url(${editorBackground})` }}>
        <Loading label={t('library.openingEditor')} />
      </div>
    )
  return (
    <div
      className="page page--editor has-bg-image"
      style={{ backgroundImage: `url(${editorBackground})` }}
    >
      <PageHeader
        eyebrow={t('library.typeTranscript', typeLabel[item.type])}
        title={t('library.reviewTranscript')}
        description={t('library.reviewDescription')}
        back={
          <button className="back-link" onClick={() => navigate(-1)}>
            {t('library.back')}
          </button>
        }
      />
      <TranscriptEditor initialTranscript={item} onTranscriptChange={setItem} />
    </div>
  )
}
