import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'
import { translate } from '../i18n/translations'

const LanguageContext = createContext(null)
const STORAGE_KEY = 'sinhaspeech_language'
const SUPPORTED = ['en', 'si']

function readLanguage() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    return SUPPORTED.includes(stored) ? stored : 'en'
  } catch {
    return 'en'
  }
}

export function LanguageProvider({ children }) {
  const [language, setLanguage] = useState(readLanguage)

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, language)
    document.documentElement.lang = language
  }, [language])

  const t = useCallback((key, ...args) => translate(key, language, ...args), [language])

  const value = useMemo(() => ({ language, setLanguage, t }), [language, t])
  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>
}

export function useLanguage() {
  return useContext(LanguageContext)
}
