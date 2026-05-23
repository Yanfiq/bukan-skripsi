const fs = require('node:fs/promises');
const path = require('node:path');
const { chromium } = require('playwright');

function parseArgs(argv) {
  const args = {
    input: 'texts.txt',
    output: 'translations.txt',
    failed: 'failed_translations.txt',
    url: 'https://www.deepl.com/en/translator/l/en/id',
    headless: true,
    timeoutMs: 60000,
    minDelayMs: 1000,
    maxDelayMs: 2000,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const current = argv[index];

    if (current === '--input' && argv[index + 1]) {
      args.input = argv[++index];
      continue;
    }

    if (current === '--output' && argv[index + 1]) {
      args.output = argv[++index];
      continue;
    }

    if (current === '--failed' && argv[index + 1]) {
      args.failed = argv[++index];
      continue;
    }

    if (current === '--url' && argv[index + 1]) {
      args.url = argv[++index];
      continue;
    }

    if (current === '--timeout' && argv[index + 1]) {
      args.timeoutMs = Number(argv[++index]);
      continue;
    }

    if (current === '--min-delay' && argv[index + 1]) {
      args.minDelayMs = Number(argv[++index]);
      continue;
    }

    if (current === '--max-delay' && argv[index + 1]) {
      args.maxDelayMs = Number(argv[++index]);
      continue;
    }

    if (current === '--headed') {
      args.headless = false;
      continue;
    }

    if (current === '--headless') {
      args.headless = true;
    }
  }

  return args;
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function randomDelay(minMs, maxMs) {
  const lower = Math.min(minMs, maxMs);
  const upper = Math.max(minMs, maxMs);
  return Math.floor(lower + Math.random() * (upper - lower + 1));
}

async function ensureParentDir(filePath) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
}

async function prepareOutputFiles(outputPath, failedPath) {
  await ensureParentDir(outputPath);
  await ensureParentDir(failedPath);

  try {
    await fs.access(outputPath);
  } catch (error) {
    await fs.writeFile(outputPath, '', 'utf8');
  }

  try {
    await fs.access(failedPath);8
  } catch (error) {
    await fs.writeFile(failedPath, '', 'utf8');
  }
}

async function dismissCookies(page) {
  const closeButton = page.getByTestId('cookie-banner-lax-close-button');
  if (await closeButton.count()) {
    await closeButton.first().click({ timeout: 5000 }).catch(() => {});
  }
}

async function translateOnce(page, text, url, timeoutMs) {
  // Assumes `page` is already at the translator URL and cookies dismissed.
  await dismissCookies(page);

  const textboxes = page.locator('main [role="textbox"]');
  const sourceBox = textboxes.first();
  const targetBox = textboxes.nth(1);

  await sourceBox.waitFor({ state: 'visible', timeout: timeoutMs });
  await sourceBox.fill(text);

  await page.waitForFunction(
    () => {
      const textboxes = document.querySelectorAll('main [role="textbox"]');
      const targetBox = textboxes[1];
      return targetBox && targetBox.innerText && targetBox.innerText.trim().length > 0;
    },
    { timeout: timeoutMs }
  );

  const result = (await targetBox.innerText()).trim();

  // Try to press the clear button (appears after input) so next translation can start fresh.
  await clearSource(page).catch(() => {});

  return result;
}

async function clearSource(page) {
  // Try several selectors for DeepL's clear/source-reset button, then fall back to clearing the source textbox.
  const selectors = [
    'button[data-testid="lmt__clear_text_button"]',
    'button[aria-label*="Clear"]',
    'button[aria-label*="clear"]',
    '.lmt__clear_text_button',
    'button[title*="Clear"]'
  ];

  for (const sel of selectors) {
    try {
      const el = page.locator(sel);
      if (await el.count()) {
        await el.first().click({ timeout: 3000 }).catch(() => {});
        return;
      }
    } catch (e) {
      // ignore and try next
    }
  }

  // Fallback: clear the source textbox directly
  try {
    const src = page.locator('main [role="textbox"]').first();
    await src.fill('');
  } catch (e) {
    // give up
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const inputPath = path.resolve(process.cwd(), args.input);
  const outputPath = path.resolve(process.cwd(), args.output);
  const failedPath = path.resolve(process.cwd(), args.failed);

  const rawInput = await fs.readFile(inputPath, 'utf8');
  const texts = rawInput.split(/\r?\n/);

  await prepareOutputFiles(outputPath, failedPath);

  let browser = await chromium.launch({ headless: args.headless });
  let page = await browser.newPage({ viewport: { width: 1440, height: 1200 } });

  // Navigate to DeepL once at startup
  await page.goto(args.url, { waitUntil: 'domcontentloaded', timeout: args.timeoutMs }).catch(() => {});
  await page.waitForLoadState('networkidle', { timeout: args.timeoutMs }).catch(() => {});
  await dismissCookies(page);

  async function restartBrowser(oldBrowser, headless) {
    try {
      if (oldBrowser) await oldBrowser.close();
    } catch (e) {
      // ignore errors closing old browser
    }

    const newBrowser = await chromium.launch({ headless });
    const newPage = await newBrowser.newPage({ viewport: { width: 1440, height: 1200 } });
    return { newBrowser, newPage };
  }

  try {
    console.log(`Loaded ${texts.length} lines from ${inputPath}`);
    console.log(`Writing translations to ${outputPath}`);

    for (const [index, originalText] of texts.entries()) {
      const text = originalText.trim();

      if (!text) {
        await fs.appendFile(outputPath, '\n', 'utf8');
        continue;
      }

      let translatedText = null;
      let lastError = null;

      for (let attempt = 1; attempt <= 3; attempt += 1) {
        try {
          console.log(`[${index + 1}/${texts.length}] Translating (attempt ${attempt}): ${text}`);
          translatedText = await translateOnce(page, text, args.url, args.timeoutMs);
          break;
        } catch (error) {
          lastError = error;
          console.error(`Failed on attempt ${attempt}: ${error.message}`);
          // Attempt to clear state by restarting browser/page before next attempt
          if (attempt < 3) {
            console.log('Clearing cookies/cache by restarting browser and creating a fresh context...');
            const res = await restartBrowser(browser, args.headless);
            browser = res.newBrowser;
            page = res.newPage;
            // navigate to the translator page on the fresh page
            await page.goto(args.url, { waitUntil: 'domcontentloaded', timeout: args.timeoutMs }).catch(() => {});
            await page.waitForLoadState('networkidle', { timeout: args.timeoutMs }).catch(() => {});
            await dismissCookies(page);
            // give the new page a moment to be ready
            await delay(1000);
          }

          await delay(1500);
        }
      }

      if (translatedText === null) {
        console.error(`Giving up on line ${index + 1}`);
        await fs.appendFile(failedPath, `${originalText}\n`, 'utf8');
        if (lastError) {
          console.error(lastError);
        }
        continue;
      }

      console.log(`-> ${translatedText}`);
      await fs.appendFile(outputPath, `${translatedText}\n`, 'utf8');
      await delay(randomDelay(args.minDelayMs, args.maxDelayMs));
    }
  } finally {
    try {
      if (browser) await browser.close();
    } catch (e) {
      // ignore close errors
    }
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});