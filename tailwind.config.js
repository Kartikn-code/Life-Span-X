/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        navy:    '#0A0E1A',
        teal:    '#00F5D4',
        amber:   '#F5A623',
        danger:  '#FF4D4F',
        surface: '#111827',
        card:    '#1F2937',
        border:  '#374151',
      },
      fontFamily: {
        display: ['Sora', 'sans-serif'],
        body: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      }
    },
  },
  plugins: [],
}
