
---

## 1. Scaffold your Vite React app

```bash
npm create vite@latest my-app
cd my-app
npm install
```

* Choose **React** → **JavaScript**.

---

## 2. Install Tailwind & PostCSS

```bash
npm install -D tailwindcss@3 postcss autoprefixer
npx tailwindcss init -p
```

This generates `tailwind.config.js` and `postcss.config.js`.

---

## 3. Configure Tailwind

**tailwind.config.js**

```js
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx}',
  ],
  theme: { extend: {} },
  plugins: [],
}
```

**postcss.config.js**

```js
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

---

## 4. Set up your CSS entry

**src/index.css**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

And import it in your entrypoint:

**src/main.jsx** (or `index.jsx`)

```js
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

```

---

## 5. Configure path aliases (optional, but handy)

**vite.config.js**

```js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
})
```

And in **jsconfig.json** (for your editor):

```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": { "@/*": ["src/*"] }
  },
  "exclude": ["node_modules", "dist"]
}
```

---

## 6. Initialize shadcn/ui

```bash
npx shadcn@latest init
```

* style : Default
* Base color: choose “Neutral” or whatever you like

---

## 7. Add components

For example, to add a Button component:

```bash
npx shadcn@latest add button
```

Then use it:

```jsx
// src/App.jsx
import { Button } from "@/components/ui/button"

function App() {
  return (
    <div className="flex min-h-svh flex-col items-center justify-center">
      <Button>Click me</Button>
    </div>
  )
}

export default App
```

---

## 8. Run the dev server

```bash
npm run dev
```

Visit \*\*[http://localhost:5173\*\*—you](http://localhost:5173**—you) should see your Tailwind‑styled React app with shadcn components ready to go!
