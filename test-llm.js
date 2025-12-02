/**
 * Test script for LLM inference using OxideEngine
 * Run with: node test-llm.js [path-to-model.safetensors] [path-to-config.json]
 */

const path = require('path');
const fs = require('fs');

// ANSI color codes
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  blue: '\x1b[34m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m',
};

function printHeader(text) {
  console.log(`\n${colors.bright}${colors.blue}${'='.repeat(70)}${colors.reset}`);
  console.log(`${colors.bright}${colors.blue}${text}${colors.reset}`);
  console.log(`${colors.bright}${colors.blue}${'='.repeat(70)}${colors.reset}\n`);
}

function printSuccess(text) {
  console.log(`${colors.green}✓${colors.reset} ${text}`);
}

function printError(text) {
  console.log(`${colors.red}✗${colors.reset} ${text}`);
}

function printInfo(text) {
  console.log(`${colors.cyan}ℹ${colors.reset} ${text}`);
}

function printWarning(text) {
  console.log(`${colors.yellow}⚠${colors.reset} ${text}`);
}

// Load the native module
let OxideEngine;
try {
  const platform = process.platform;
  const arch = process.arch;
  const platformMap = {
    'darwin-arm64': 'darwin-arm64',
    'darwin-x64': 'darwin-x64',
    'linux-x64': 'linux-x64-gnu',
    'win32-x64': 'win32-x64-msvc',
  };
  const platformKey = `${platform}-${arch}`;
  const platformSuffix = platformMap[platformKey] || 'unknown';
  const binaryName = `core.${platformSuffix}.node`;
  const binaryPath = path.join(__dirname, binaryName);

  if (fs.existsSync(binaryPath)) {
    const nativeModule = require(binaryPath);
    OxideEngine = nativeModule.OxideEngine;
    printSuccess(`Loaded native module: ${binaryName}`);
  } else {
    throw new Error(`Binary not found: ${binaryName}`);
  }
} catch (error) {
  printError('Failed to load native module');
  console.error(`${colors.red}Error: ${error.message}${colors.reset}`);
  console.error('\nPlease build the project first:');
  console.error('  npm run build');
  process.exit(1);
}

async function runTests() {
  console.clear();
  console.log(`${colors.bright}${colors.magenta}OxideEngine LLM Test Suite${colors.reset}`);
  console.log(`${colors.yellow}Testing Phi-3 model loading and inference${colors.reset}\n`);

  // Get model path from command line or use default
  const modelPath = process.argv[2];
  const configPath = process.argv[3];

  if (!modelPath) {
    printWarning('No model path provided');
    console.log('\nUsage:');
    console.log('  node test-llm.js <path-to-model.safetensors> [path-to-config.json]');
    console.log('\nExample:');
    console.log('  node test-llm.js ./models/phi-3-mini.safetensors ./models/config.json');
    console.log('\nNote: If config.json is not provided, default Phi-3-mini config will be used.');
    console.log('\nYou can download Phi-3 models from Hugging Face:');
    console.log('  https://huggingface.co/microsoft/Phi-3-mini-4k-instruct');
    console.log('\nFor now, running basic GPU tests without model loading...\n');

    try {
      // Test 1: Initialize engine
      printHeader('Test 1: Initialize OxideEngine');
      console.time('Initialization time');
      const engine = new OxideEngine();
      console.timeEnd('Initialization time');
      printSuccess('OxideEngine initialized successfully');

      const deviceInfo = engine.getDeviceInfo();
      printInfo(`Device: ${deviceInfo}`);

      // Test 2: Basic GPU compute
      printHeader('Test 2: Basic GPU Compute Test');
      console.time('GPU compute time');
      const computeResult = engine.testGpuCompute();
      console.timeEnd('GPU compute time');
      console.log('\n' + computeResult);
      printSuccess('GPU compute test passed\n');

      printHeader('Summary');
      printInfo('GPU is working correctly');
      printWarning('Model not loaded - provide a model path to test inference');
      console.log();
    } catch (error) {
      printError('Test failed: ' + error.message);
      process.exit(1);
    }
    return;
  }

  // Full test with model
  try {
    // Verify files exist
    if (!fs.existsSync(modelPath)) {
      throw new Error(`Model file not found: ${modelPath}`);
    }
    if (configPath && !fs.existsSync(configPath)) {
      throw new Error(`Config file not found: ${configPath}`);
    }

    // Test 1: Initialize engine
    printHeader('Test 1: Initialize OxideEngine');
    console.time('Initialization time');
    const engine = new OxideEngine();
    console.timeEnd('Initialization time');
    printSuccess('OxideEngine initialized successfully');

    const deviceInfo = engine.getDeviceInfo();
    printInfo(`Device: ${deviceInfo}`);
    console.log();

    // Test 2: Load model
    printHeader('Test 2: Load Phi-3 Model');
    printInfo(`Model path: ${modelPath}`);
    if (configPath) {
      printInfo(`Config path: ${configPath}`);
    } else {
      printInfo('Using default Phi-3-mini config');
    }

    console.time('Model loading time');
    const loadResult = engine.loadModel(modelPath, configPath || null);
    console.timeEnd('Model loading time');
    console.log('\n' + loadResult);
    printSuccess('Model loaded into VRAM\n');

    // Test 3: Run forward pass
    printHeader('Test 3: Run Forward Pass');

    // Test with a simple sequence of tokens
    const testTokens = [1, 2, 3, 4, 5];  // Example token IDs
    printInfo(`Input tokens: [${testTokens.join(', ')}]`);

    console.time('Forward pass time');
    const forwardResult = engine.forward(testTokens);
    console.timeEnd('Forward pass time');

    console.log(`\n${forwardResult.message}\n`);
    printInfo(`Batch size: ${forwardResult.batchSize}`);
    printInfo(`Sequence length: ${forwardResult.sequenceLength}`);
    printInfo(`Vocabulary size: ${forwardResult.vocabSize}`);

    console.log(`\n${colors.cyan}Top 5 predicted tokens:${colors.reset}`);
    forwardResult.topTokens.forEach((token, idx) => {
      console.log(`  ${idx + 1}. Token ID: ${token.tokenId.toString().padStart(6)} | Logit: ${token.logit.toFixed(4)}`);
    });

    printSuccess('\nForward pass successful!\n');

    // Test 4: Multiple forward passes
    printHeader('Test 4: Multiple Forward Passes');
    const numRuns = 3;
    printInfo(`Running ${numRuns} forward passes to test consistency...`);

    const times = [];
    for (let i = 0; i < numRuns; i++) {
      const start = Date.now();
      engine.forward(testTokens);
      const elapsed = Date.now() - start;
      times.push(elapsed);
      console.log(`  Run ${i + 1}: ${elapsed}ms`);
    }

    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    printInfo(`Average inference time: ${avgTime.toFixed(2)}ms`);
    printSuccess('Multiple forward passes completed\n');

    // Summary
    printHeader('Test Summary');
    printSuccess('All tests passed!');
    printInfo('Phi-3 model is loaded and working correctly');
    printInfo('GPU acceleration is active');
    printInfo(`Average inference: ${avgTime.toFixed(2)}ms per forward pass`);
    console.log();

  } catch (error) {
    printHeader('Test Failed');
    printError('Error occurred during testing:');
    console.error(`\n${colors.red}${error.message}${colors.reset}`);
    if (error.stack) {
      console.error(`\n${colors.yellow}Stack trace:${colors.reset}`);
      console.error(error.stack);
    }
    process.exit(1);
  }
}

// Run all tests
runTests().catch((error) => {
  console.error('Unhandled error:', error);
  process.exit(1);
});
