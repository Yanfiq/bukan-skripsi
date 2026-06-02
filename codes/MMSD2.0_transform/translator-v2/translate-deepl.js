const fs = require('node:fs/promises');
const path = require('node:path');
const { chromium } = require('playwright');

function parseArgs(argv) {
  const args = {
    input: 'texts.txt',
    output: 'translations.txt',
    failed: 'failed_translations.txt',
    inputFormat: null,
    outputFormat: null,
    jsonInputKey: null,
    jsonOutputKey: 'translation',
    repairDuplicates: false,
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

    if (current === '--input-format' && argv[index + 1]) {
      args.inputFormat = argv[++index];
      continue;
    }

    if (current === '--output-format' && argv[index + 1]) {
      args.outputFormat = argv[++index];
      continue;
    }

    if (current === '--json-input-key' && argv[index + 1]) {
      args.jsonInputKey = argv[++index];
      continue;
    }

    if (current === '--json-output-key' && argv[index + 1]) {
      args.jsonOutputKey = argv[++index];
      continue;
    }

    if (current === '--repair-duplicates') {
      args.repairDuplicates = true;
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

async function prepareOutputFiles(outputPath, failedPath, outputFormat = 'txt') {
  await ensureParentDir(outputPath);
  await ensureParentDir(failedPath);

  try {
    await fs.access(outputPath);
  } catch (error) {
    if (outputFormat === 'json') {
      // Start a JSON array but don't close it; we'll append items incrementally.
      await fs.writeFile(outputPath, '[\n', 'utf8');
    } else {
      await fs.writeFile(outputPath, '', 'utf8');
    }
  }

  try {
    await fs.access(failedPath);
  } catch (error) {
    await fs.writeFile(failedPath, '', 'utf8');
  }
}

function inferFormat(filePath, explicitFormat) {
  if (explicitFormat) {
    return explicitFormat.toLowerCase();
  }

  const ext = path.extname(filePath).toLowerCase();
  if (ext === '.json') return 'json';
  return 'txt';
}

function getByPath(value, keyPath) {
  if (!keyPath) return value;
  return keyPath.split('.').reduce((current, key) => (current == null ? undefined : current[key]), value);
}

function setByPath(value, keyPath, newValue) {
  if (!keyPath) return newValue;

  const parts = keyPath.split('.');
  const root = value && typeof value === 'object' ? (Array.isArray(value) ? [...value] : { ...value }) : {};
  let current = root;

  for (let index = 0; index < parts.length - 1; index += 1) {
    const part = parts[index];
    const next = current[part];
    current[part] = Array.isArray(next) ? [...next] : { ...(next || {}) };
    current = current[part];
  }

  current[parts[parts.length - 1]] = newValue;
  return root;
}

async function readInputData(inputPath, inputFormat, jsonInputKey) {
  if (inputFormat === 'json') {
    const raw = await fs.readFile(inputPath, 'utf8');
    const parsed = JSON.parse(raw);

    if (Array.isArray(parsed)) {
      return parsed.map((item, index) => {
        const itemType = item === null ? 'null' : typeof item;
        if (!jsonInputKey && itemType === 'object') {
          throw new Error('JSON array items are objects; use --json-input-key to choose the text field to translate');
        }

        const text = jsonInputKey ? getByPath(item, jsonInputKey) : item;
        return {
          type: 'json',
          index,
          item,
          text: text == null ? '' : String(text),
        };
      });
    }

    if (parsed && typeof parsed === 'object') {
      return Object.entries(parsed).map(([key, item]) => {
        const itemType = item === null ? 'null' : typeof item;
        if (!jsonInputKey && itemType === 'object') {
          throw new Error(`JSON value for key "${key}" is an object; use --json-input-key to choose the text field to translate`);
        }

        const text = jsonInputKey ? getByPath(item, jsonInputKey) : item;
        return {
          type: 'json-object',
          key,
          item,
          text: text == null ? '' : String(text),
        };
      });
    }

    throw new Error('JSON input must be an array or object');
  }

  const rawInput = await fs.readFile(inputPath, 'utf8');
  return rawInput.split(/\r?\n/).map((text, index) => ({
    type: 'txt',
    index,
    text,
  }));
}

async function readExistingJsonOutput(outputPath) {
  const raw = await fs.readFile(outputPath, 'utf8').catch(() => '');
  if (!raw.trim()) {
    return null;
  }

  return JSON.parse(raw);
}

function extractStoredTranslation(entry, jsonOutputKey) {
  if (entry == null) {
    return null;
  }

  if (typeof entry !== 'object') {
    return String(entry);
  }

  const value = getByPath(entry, jsonOutputKey);
  return value == null ? null : String(value);
}

async function writeJsonOutput(outputPath, inputData, translations, jsonOutputKey) {
  const existing = await fs.readFile(outputPath, 'utf8').catch(() => '');
  const parsedExisting = existing.trim() ? JSON.parse(existing) : null;

  if (inputData.length && inputData[0].type === 'json') {
    const output = inputData.map((entry, index) => {
      const translatedText = translations[index];
      return setByPath(entry.item, jsonOutputKey, translatedText);
    });

    await fs.writeFile(outputPath, `${JSON.stringify(output, null, 2)}\n`, 'utf8');
    return;
  }

  if (inputData.length && inputData[0].type === 'json-object') {
    const output = {};
    inputData.forEach((entry, index) => {
      output[entry.key] = setByPath(entry.item, jsonOutputKey, translations[index]);
    });

    await fs.writeFile(outputPath, `${JSON.stringify(output, null, 2)}\n`, 'utf8');
    return;
  }

  if (parsedExisting && Array.isArray(parsedExisting)) {
    await fs.writeFile(outputPath, `${JSON.stringify([...parsedExisting, ...translations], null, 2)}\n`, 'utf8');
    return;
  }

  await fs.writeFile(outputPath, `${JSON.stringify(translations, null, 2)}\n`, 'utf8');
}

async function writeJsonIncremental(outputPath, entry, index, translatedText, jsonOutputKey) {
  const value = setByPath(entry.item, jsonOutputKey, translatedText);
  await appendPrettyJsonValue(outputPath, value);
}

async function appendPrettyJsonValue(outputPath, value) {
  // Efficient incremental append: only read a small tail of the file to decide whether
  // to replace the trailing newline with a comma/newline or append after the opening bracket.
  const fh = await fs.open(outputPath, 'r+');
  try {
    const st = await fh.stat();
    const size = st.size;
    const lastChunkSize = Math.min(1024, Math.max(1, size));
    let lastChar = null;
    const prettyItem = JSON.stringify(value, null, 2)
      .split('\n')
      .map((line) => `  ${line}`)
      .join('\n');

    if (size === 0) {
      // Shouldn't happen because prepareOutputFiles should write '[\n' for JSON, but handle anyway.
      await fh.write(Buffer.from(`[\n${prettyItem}\n`));
      return;
    }

    const readBuffer = Buffer.alloc(lastChunkSize);
    await fh.read(readBuffer, 0, lastChunkSize, size - lastChunkSize);
    // find last non-whitespace character
    for (let i = readBuffer.length - 1; i >= 0; i -= 1) {
      const ch = String.fromCharCode(readBuffer[i]);
      if (!/\s/.test(ch)) {
        lastChar = ch;
        // compute absolute position of this char
        const posOfLast = size - lastChunkSize + i;
        if (lastChar === ']') {
          // overwrite the trailing ']' with a comma/newline and then append item (leaving file unclosed)
          const payload = `,\n${prettyItem}\n`;
          await fh.write(Buffer.from(payload), 0, Buffer.byteLength(payload), posOfLast);
          return;
        }

        // otherwise, replace the last newline with comma/newline and append the item.
        if (lastChar === '[') {
          const payload = `${prettyItem}\n`;
          await fh.write(Buffer.from(payload), 0, Buffer.byteLength(payload), size);
          return;
        }

        const payload = `,\n${prettyItem}\n`;
        await fh.write(Buffer.from(payload), 0, Buffer.byteLength(payload), size - 1);
        return;
      }
    }

    // if we couldn't find a non-whitespace char in the tail, just append item
    const payload = `${prettyItem}\n`;
    await fh.write(Buffer.from(payload), 0, Buffer.byteLength(payload), size);
  } finally {
    await fh.close();
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
  const inputFormat = inferFormat(inputPath, args.inputFormat);
  const outputFormat = inferFormat(outputPath, args.outputFormat);
  const repairDuplicates = Boolean(args.repairDuplicates && outputFormat === 'json');

  const inputData = await readInputData(inputPath, inputFormat, args.jsonInputKey);
  const existingJsonOutput = repairDuplicates ? await readExistingJsonOutput(outputPath) : null;

  if (repairDuplicates) {
    await ensureParentDir(outputPath);
    await fs.writeFile(outputPath, '[\n', 'utf8');
  } else {
    await prepareOutputFiles(outputPath, failedPath, outputFormat);
  }

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
    console.log(`Loaded ${inputData.length} items from ${inputPath} (${inputFormat})`);
    console.log(`Writing translations to ${outputPath} (${outputFormat})`);
    if (repairDuplicates) {
      console.log('Repair mode enabled: duplicate translation blocks will be retranslated.');
    }

    let lastStoredTranslation = null;

    for (const [index, entry] of inputData.entries()) {
      const originalText = entry.text ?? '';
      const text = String(originalText).trim();

      const storedEntry = repairDuplicates
        ? (Array.isArray(existingJsonOutput)
          ? existingJsonOutput[index]
          : existingJsonOutput && typeof existingJsonOutput === 'object'
            ? existingJsonOutput[entry.key]
            : undefined)
        : undefined;

      const storedTranslation = repairDuplicates ? extractStoredTranslation(storedEntry, args.jsonOutputKey) : null;
      const isDuplicate = repairDuplicates
        && storedTranslation != null
        && lastStoredTranslation != null
        && storedTranslation === lastStoredTranslation;

      if (repairDuplicates && storedEntry !== undefined && !isDuplicate) {
        await appendPrettyJsonValue(outputPath, storedEntry);
        lastStoredTranslation = storedTranslation;
        continue;
      }

      if (!text) {
        if (outputFormat === 'txt') {
          await fs.appendFile(outputPath, '\n', 'utf8');
        } else {
          await writeJsonIncremental(outputPath, entry, index, null, args.jsonOutputKey);
        }
        continue;
      }

      let translatedText = null;
      let lastError = null;

      for (let attempt = 1; attempt <= 5; attempt += 1) {
        try {
          console.log(`[${index + 1}/${inputData.length}] Translating (attempt ${attempt}): ${text}`);
          translatedText = await translateOnce(page, text, args.url, args.timeoutMs);
          break;
        } catch (error) {
          lastError = error;
          console.error(`Failed on attempt ${attempt}: ${error.message}`);
          // Attempt to clear state by restarting browser/page before next attempt
          if (attempt < 5) {
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
        console.error(`Giving up on item ${index + 1}`);
        await fs.appendFile(failedPath, `${text}\n`, 'utf8');
        if (lastError) {
          console.error(lastError);
        }
        continue;
      }

      console.log(`-> ${translatedText}`);
      if (outputFormat === 'txt') {
        await fs.appendFile(outputPath, `${translatedText}\n`, 'utf8');
      } else {
        await writeJsonIncremental(outputPath, entry, index, translatedText, args.jsonOutputKey);
      }

      if (repairDuplicates) {
        lastStoredTranslation = translatedText;
      }

      await delay(randomDelay(args.minDelayMs, args.maxDelayMs));
    }

    // JSON output is written incrementally per item via writeJsonIncremental
  } finally {
    try {
      if (browser) await browser.close();
    } catch (e) {
      // ignore close errors
    }
    // finalize JSON output if needed
    try {
      if (outputFormat === 'json') {
        await finalizeOutputJson(outputPath).catch(() => {});
      }
    } catch (e) {
      // ignore
    }
  }
}

async function finalizeOutputJson(outputPath) {
  try {
    const fh = await fs.open(outputPath, 'a+');
    try {
      const st = await fh.stat();
      const size = st.size;
      if (size === 0) {
        await fh.write('[]\n');
        return;
      }

      const lastChunkSize = Math.min(1024, size);
      const buf = Buffer.alloc(lastChunkSize);
      await fh.read(buf, 0, lastChunkSize, size - lastChunkSize);
      let lastChar = null;
      for (let i = buf.length - 1; i >= 0; i -= 1) {
        const ch = String.fromCharCode(buf[i]);
        if (!/\s/.test(ch)) {
          lastChar = ch;
          break;
        }
      }

      if (lastChar === ']') {
        return; // already closed
      }

      // append closing bracket
      await fh.write(']\n');
    } finally {
      await fh.close();
    }
  } catch (e) {
    // ignore finalize errors
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});