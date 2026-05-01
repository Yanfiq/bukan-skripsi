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
    minDelayMs: 2500,
    maxDelayMs: 5000,
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
  await fs.writeFile(outputPath, '', 'utf8');
  await fs.writeFile(failedPath, '', 'utf8');
}

async function dismissCookies(page) {
  const closeButton = page.getByTestId('cookie-banner-lax-close-button');
  if (await closeButton.count()) {
    await closeButton.first().click({ timeout: 5000 }).catch(() => {});
  }
}

async function translateOnce(page, text, url, timeoutMs) {
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: timeoutMs });
  await page.waitForLoadState('networkidle', { timeout: timeoutMs }).catch(() => {});
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

  return (await targetBox.innerText()).trim();
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const inputPath = path.resolve(process.cwd(), args.input);
  const outputPath = path.resolve(process.cwd(), args.output);
  const failedPath = path.resolve(process.cwd(), args.failed);

  const rawInput = await fs.readFile(inputPath, 'utf8');
  const texts = rawInput.split(/\r?\n/);

  await prepareOutputFiles(outputPath, failedPath);

  const browser = await chromium.launch({ headless: args.headless });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1200 } });

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
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});