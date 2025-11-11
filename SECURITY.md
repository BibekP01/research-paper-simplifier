# API Key Security

This document explains how to securely manage your Google Gemini API key in this project.

## Getting a Google Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key

## Setting Up Your Environment

1. **Never commit your API key to version control**
   - The `.env` file is already in `.gitignore`
   - Double-check that your `.env` file is not being tracked by Git:
     ```bash
     git check-ignore .env
     ```

2. **Create a `.env` file**
   ```bash
   cp .env.example .env
   ```

3. **Edit the `.env` file**
   ```bash
   # Google Gemini API Key (required)
   GEMINI_API_KEY=your-api-key-here
   
   # Optional overrides (defaults are in config/config.yaml)
   # GEMINI_MODEL=gemini-2.5-flash
   # GEMINI_TEMPERATURE=0.3
   # GEMINI_MAX_TOKENS=2048
   ```

4. **Verify your setup**
   ```bash
   python -c "from dotenv import load_dotenv; load_dotenv(); print('API Key loaded successfully' if 'GEMINI_API_KEY' in os.environ else 'API Key not found')"
   ```

## If Your API Key is Compromised

If you accidentally committed an API key to GitHub:

1. **Immediately revoke the exposed key** in Google AI Studio
2. **Rotate the key** (create a new one and update your `.env` file)
3. **Clean your Git history** if the key was committed:
   ```bash
   # Remove the key from all commits
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch .env" \
     --prune-empty --tag-name-filter cat -- --all
   
   # Force push the changes
   git push origin --force --all
   ```

## Best Practices

- **Never** share your API key in public forums or screenshots
- Use environment variables instead of hardcoding the key
- Consider using a secrets manager for production deployments
- Rotate your API keys periodically
- Set up usage alerts in Google Cloud Console

## Testing with a Different Key

For testing, you can create a `.env.test` file:

```bash
# .env.test
GEMINI_API_KEY=test-api-key-here
```

Then load it specifically for tests:
```python
from dotenv import load_dotenv
load_dotenv('.env.test')  # Override with test environment
```
