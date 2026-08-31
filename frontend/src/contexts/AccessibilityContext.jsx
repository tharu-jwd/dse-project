import { createContext, useContext, useEffect, useMemo, useState } from 'react'

const AccessibilityContext = createContext(null)
const defaults = {
  fontSize: 'normal',
  highContrast: false,
  confidenceThreshold: 0.8,
  interactionMode: 'normal', // 'normal' (keyboard & mouse) | 'command' (voice-command driven)
}

function readPreferences() {
  try {
    return { ...defaults, ...JSON.parse(localStorage.getItem('sinhaspeech_accessibility')) }
  } catch {
    return defaults
  }
}

export function AccessibilityProvider({ children }) {
  const [preferences, setPreferences] = useState(readPreferences)
  const updatePreference = (key, value) =>
    setPreferences((current) => ({ ...current, [key]: value }))

  useEffect(() => {
    localStorage.setItem('sinhaspeech_accessibility', JSON.stringify(preferences))
    document.documentElement.dataset.contrast = preferences.highContrast ? 'high' : 'normal'
    document.documentElement.dataset.transcriptSize = preferences.fontSize
  }, [preferences])

  const value = useMemo(() => ({ ...preferences, updatePreference }), [preferences])
  return <AccessibilityContext.Provider value={value}>{children}</AccessibilityContext.Provider>
}

export function useAccessibility() {
  return useContext(AccessibilityContext)
}
