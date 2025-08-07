# 📁 Salsa block monitoring files

Essential files needed to run the salsa block monitoring system.

## 🌳 File Structure

### Minimal Setup (Console Only)
```
salsa-block-monitor/
├── tacoclicker_mempool_monitor.py    # Main monitoring script
├── salsa_block_analyzer.py           # Salsa block analysis engine
├── bitcoin_rpc_credentials.json      # Bitcoin Core RPC settings
└── Log/                              # Log files directory (auto-created)
```

### Full Setup (With Telegram)
```
salsa-block-monitor/
├── 🔧 Core Scripts
│   ├── tacoclicker_mempool_monitor.py    # Main monitoring script
│   ├── salsa_block_analyzer.py           # Salsa block analysis engine
│   └── simple_telegram_bot.py            # Telegram notification system
│
├── ⚙️ Configuration Files
│   ├── bitcoin_rpc_credentials.json      # Bitcoin Core RPC settings
│   ├── telegram_config.json             # Telegram bot token
│   └── telegram_subscribers.json        # Subscriber list (auto-created)
│
└── 📊 Data & Logs
    └── Log/                              # Log files directory (auto-created)
        ├── mempool_monitor_*.log         # Main system logs
        ├── bet_transactions_*.log        # Detected bets
        └── salsa_analysis_*.log          # Salsa block analysis
```

## 📋 Required Files Details

### 🔧 Core Python Scripts

| File | Purpose | Required For |
|------|---------|--------------|
| `tacoclicker_mempool_monitor.py` | Main monitoring system | ✅ Always |
| `salsa_block_analyzer.py` | Salsa block analysis | ✅ Always |
| `simple_telegram_bot.py` | Telegram notifications | 🔶 Only with --telegram |

### ⚙️ Configuration Files

#### `bitcoin_rpc_credentials.json` ✅ **Required**
```json
{
  "rpchost": "127.0.0.1",
  "rpcport": 8332,
  "rpcuser": "your_rpc_username",
  "rpcpassword": "your_rpc_password"
}
```

#### `telegram_config.json` 🔶 **Required for Telegram**
```json
{
  "bot_token": "1234567890:ABCdefGHIjklMNOpqrsTUVwxyz"
}
```

#### `telegram_subscribers.json` 🔶 **Auto-created**
```json
{
  "subscribers": ["123456789", "987654321"]
}
```

## 🚀 Launch Commands

### Console Only
```bash
# Minimal setup - no Telegram
python tacoclicker_mempool_monitor.py
```

### With Telegram Notifications
```bash
# Full production setup
python tacoclicker_mempool_monitor.py --telegram
```

### Custom Intervals
```bash
# Mempool scan every 15s, block check every 5s
python tacoclicker_mempool_monitor.py --telegram 15 5
```

## 📦 File Dependencies

```
tacoclicker_mempool_monitor.py
├── imports: salsa_block_analyzer.py (when salsa blocks found)
├── imports: simple_telegram_bot.py (if --telegram flag used)
├── reads: bitcoin_rpc_credentials.json
├── reads: telegram_config.json (if --telegram)
├── reads: telegram_subscribers.json (if --telegram)
└── writes: Log/*.log files

salsa_block_analyzer.py
├── reads: bitcoin_rpc_credentials.json
└── writes: salsa_analysis_*.json files

simple_telegram_bot.py
├── reads: telegram_config.json
├── reads/writes: telegram_subscribers.json
└── sends: Telegram API calls
```

## 🎯 Production Checklist

### Before Launch
- [ ] Bitcoin Core running and synced
- [ ] `bitcoin_rpc_credentials.json` configured
- [ ] `telegram_config.json` configured (if using Telegram)
- [ ] Telegram bot tested and subscribers added
- [ ] `Log/` directory exists (or will be auto-created)

### Launch
- [ ] Run: `python tacoclicker_mempool_monitor.py --telegram`
- [ ] Verify console output shows block height
- [ ] Check Telegram notifications work
- [ ] Monitor `Log/` directory for log files

### Monitoring
- [ ] Check logs regularly: `tail -f Log/mempool_monitor_*.log`
- [ ] Verify salsa block detection every 144 blocks
- [ ] Monitor system resources (CPU, RAM)
- [ ] Ensure Bitcoin Core stays synced

## 📊 Generated Files

The system will automatically create:

```
Log/
├── mempool_monitor_20250807_123456.log     # Main system log
├── bet_transactions_20250807_123456.log    # Detected bets
├── salsa_analysis_20250807_123456.log      # Salsa analysis
└── salsa_analysis_block_908784.json        # Salsa results (JSON)
```

## 🔧 Troubleshooting

### Missing Files Error
```bash
# If you get import errors, ensure these files exist:
ls -la tacoclicker_mempool_monitor.py
ls -la salsa_block_analyzer.py
ls -la simple_telegram_bot.py  # Only needed with --telegram
```

### Configuration Errors
```bash
# Check configuration files exist and are valid JSON:
cat bitcoin_rpc_credentials.json | python -m json.tool
cat telegram_config.json | python -m json.tool  # If using Telegram
```

### Bitcoin Core Connection
```bash
# Test Bitcoin Core RPC manually:
bitcoin-cli getblockchaininfo
```

---

