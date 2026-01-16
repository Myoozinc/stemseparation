# Stem Separation Web Tool

A high-end web application for audio stem separation and song analysis (Key/BPM). Powered by the `tinahmbuz/audiostems` Hugging Face Space.

## 🚀 How to set up on GitHub

To deploy this tool to GitHub Pages:

1. **Create a GitHub Repository**:
   - Create a new repository on GitHub.
   - Do NOT initialize with a README (since we already have one).

2. **Push your code**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

3. **Enable GitHub Pages**:
   - Go to your repository **Settings** > **Pages**.
   - Under **Build and deployment** > **Source**, select **GitHub Actions**.
   - The included `.github/workflows/deploy.yml` will automatically build and deploy your site every time you push to `main`.

4. **Access your site**:
   - Once the Action completes, your site will be live at `https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/`

## Development

```bash
# Install dependencies
npm install

# Run locally
npm run dev

# Build for production
npm run build
```

## Features

- **AI Stem Separation**: Vocals, Bass, Drums, Kick, Snare, Hi-hat.
- **Key & BPM Detection**: Automatic song analysis.
- **Modern UI**: Glassmorphism aesthetic.
- **Custom Audio Players**: Integrated for each stem.
- **Batch Download**: Export all stems as a ZIP file.
