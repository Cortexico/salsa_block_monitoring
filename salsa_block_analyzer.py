#!/usr/bin/env python3
"""
SALSA BLOCK ANALYSIS (Runestone) Script - Bitcoin Core RPC Version
Comprehensive analysis tool for Bitcoin Alkanes/Runestones Taco Clicker salsa blocks
Analyzes blocks for Runestone transactions and calculates winners based on lowest SHA256 hash values
Uses local Bitcoin Core node via RPC for maximum performance and no rate limits
"""

import requests
import hashlib
import time
import json
import os
import logging
import threading
from datetime import datetime
from typing import List, Dict, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Strip emojis from console output (keep emojis for Telegram messages)
import sys as _sys, re as _re
_EMOJI_RE = _re.compile(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\u2600-\u26FF\u2700-\u27BF]")

class _EmojiStrippingWriter:
    def __init__(self, wrapped):
        self._wrapped = wrapped
    def write(self, s):
        try:
            self._wrapped.write(_EMOJI_RE.sub('', s))
        except Exception:
            self._wrapped.write(s)
    def flush(self):
        try:
            self._wrapped.flush()
        except Exception:
            pass

_sys.stdout = _EmojiStrippingWriter(_sys.stdout)
_sys.stderr = _EmojiStrippingWriter(_sys.stderr)

class SalsaBlockAnalyzer:
    """Comprehensive Salsa Block Analysis for Runestone transactions using Bitcoin Core RPC"""

    def __init__(self, rpc_credentials_file: str = "bitcoin_rpc_credentials.json", max_threads: int = 8):
        # Load Bitcoin Core RPC credentials
        self.rpc_credentials = self.load_rpc_credentials(rpc_credentials_file)

        # RPC connection settings
        self.rpc_url = f"http://{self.rpc_credentials['rpchost']}:{self.rpc_credentials['rpcport']}"
        self.rpc_auth = (self.rpc_credentials['rpcuser'], self.rpc_credentials['rpcpassword'])

        # Thread-safe RPC call tracking
        self.rpc_calls_made = 0
        self.rpc_lock = threading.Lock()

        # Multi-threading configuration with throttling
        self.max_threads = min(max_threads, 6)  # Limit to 6 threads max to avoid overwhelming node

        # Request throttling to prevent overwhelming Bitcoin Core
        self.request_semaphore = threading.Semaphore(self.max_threads * 2)  # Allow 2x threads worth of concurrent requests
        self.last_request_times = {}  # Track per-thread request timing
        self.min_request_interval = 0.01  # 10ms minimum between requests per thread

        # Setup connection session with connection pooling
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            pool_connections=self.max_threads,
            pool_maxsize=self.max_threads * 2,
            max_retries=retry_strategy
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Retry configuration for network issues
        self.max_retries = 3
        self.retry_delay = 0.1  # Shorter delay for local node

        # Setup logging
        self.setup_logging()

        # Test connection on initialization
        self.test_rpc_connection()

    def load_rpc_credentials(self, credentials_file: str) -> Dict:
        """Load Bitcoin Core RPC credentials from file"""
        # Try multiple possible paths for the credentials file
        possible_paths = [
            Path(credentials_file),  # Relative to current directory
            Path(__file__).parent / credentials_file,  # Relative to script directory
            Path.cwd() / credentials_file,  # Relative to current working directory
        ]

        creds_path = None
        for path in possible_paths:
            if path.exists():
                creds_path = path
                break

        if creds_path is None:
            raise FileNotFoundError(f"RPC credentials file not found in any of these locations: {[str(p) for p in possible_paths]}")

        with open(creds_path, 'r') as f:
            credentials = json.load(f)

        required_keys = ['rpcuser', 'rpcpassword', 'rpchost', 'rpcport']
        for key in required_keys:
            if key not in credentials:
                raise ValueError(f"Missing required credential: {key}")

        print(f"[OK] Loaded RPC credentials from: {creds_path}")
        return credentials

    def test_rpc_connection(self):
        """Test connection to Bitcoin Core RPC"""
        try:
            result = self.make_rpc_call("getblockchaininfo")
            if result:
                chain = result.get('chain', 'unknown')
                blocks = result.get('blocks', 0)
                print(f"[OK] Connected to Bitcoin Core ({chain}) - Current height: {blocks}")
                self.logger.info(f"Connected to Bitcoin Core ({chain}) - Current height: {blocks}")
            else:
                raise Exception("Failed to get blockchain info")
        except Exception as e:
            error_msg = f"Failed to connect to Bitcoin Core RPC: {e}"
            print(f"[ERROR] {error_msg}")
            print("Make sure Bitcoin Core is running and RPC credentials are correct")
            raise ConnectionError(error_msg)

    def setup_logging(self):
        """Setup comprehensive logging for bet transactions"""
        # Create Log directory if it doesn't exist
        log_dir = "Log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create timestamp for log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup main logger
        self.logger = logging.getLogger('SalsaAnalyzer')
        self.logger.setLevel(logging.DEBUG)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Create detailed log file handler
        log_file = os.path.join(log_dir, f"salsa_analysis_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Create bet transactions specific log
        bet_log_file = os.path.join(log_dir, f"bet_transactions_{timestamp}.log")
        self.bet_file_handler = logging.FileHandler(bet_log_file, encoding='utf-8')
        self.bet_file_handler.setLevel(logging.INFO)

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        bet_formatter = logging.Formatter('%(message)s')

        # Set formatters
        file_handler.setFormatter(detailed_formatter)
        self.bet_file_handler.setFormatter(bet_formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)

        # Create separate logger for bet transactions
        self.bet_logger = logging.getLogger('BetTransactions')
        self.bet_logger.setLevel(logging.INFO)
        self.bet_logger.handlers.clear()
        self.bet_logger.addHandler(self.bet_file_handler)

        self.logger.info("=== SALSA BLOCK ANALYSIS LOGGING STARTED (Bitcoin Core RPC) ===")
        self.bet_logger.info("=== BET TRANSACTIONS LOG ===")
        self.bet_logger.info("Format: TXID | Position | Fee | Size | Sender | Candidate Hash | Rank")
        self.bet_logger.info("=" * 120)

    def make_rpc_call(self, method: str, params: List = None) -> Dict:
        """Make a Bitcoin Core RPC call with connection pooling and throttling"""
        if params is None:
            params = []

        # Acquire semaphore to limit concurrent requests
        with self.request_semaphore:
            # Thread-safe call ID generation
            with self.rpc_lock:
                call_id = self.rpc_calls_made
                self.rpc_calls_made += 1

            # Per-thread request throttling
            thread_id = threading.current_thread().ident
            now = time.time()

            if thread_id in self.last_request_times:
                time_since_last = now - self.last_request_times[thread_id]
                if time_since_last < self.min_request_interval:
                    time.sleep(self.min_request_interval - time_since_last)

            self.last_request_times[thread_id] = time.time()

            payload = {
                "jsonrpc": "1.0",
                "id": f"salsa_analyzer_{call_id}_{thread_id}",
                "method": method,
                "params": params
            }

            for attempt in range(self.max_retries):
                try:
                    response = self.session.post(
                        self.rpc_url,
                        json=payload,
                        auth=self.rpc_auth,
                        timeout=20  # Longer timeout for stability
                    )

                    if response.status_code == 200:
                        try:
                            result = response.json()
                        except json.JSONDecodeError as e:
                            raise Exception(f"Invalid JSON response: {e}")

                        if 'error' in result and result['error'] is not None:
                            error_msg = result['error']
                            if isinstance(error_msg, dict):
                                error_msg = f"Code {error_msg.get('code', 'unknown')}: {error_msg.get('message', 'unknown error')}"
                            raise Exception(f"RPC error: {error_msg}")

                        if 'result' not in result:
                            raise Exception(f"No result field in RPC response")

                        return result['result']
                    else:
                        raise Exception(f"HTTP error {response.status_code}: {response.text[:200]}")

                except Exception as e:
                    if "10048" in str(e) or "socket" in str(e).lower():
                        # Socket exhaustion - wait longer
                        time.sleep(0.1)
                    elif "timeout" in str(e).lower() or "connection" in str(e).lower():
                        # Connection issues - wait and retry
                        time.sleep(0.05 * (attempt + 1))  # Exponential backoff
                    elif attempt < self.max_retries - 1:
                        # Other errors - shorter retry delay
                        time.sleep(0.02 * (attempt + 1))
                    else:
                        # Log the final failure for debugging
                        self.logger.error(f"RPC call '{method}' with params {params} failed after {self.max_retries} attempts: {e}")
                        return None  # Return None instead of raising exception to allow graceful handling

    def get_rpc_usage_stats(self) -> str:
        """Get RPC usage statistics"""
        return f"RPC Calls: {self.rpc_calls_made} (Local Bitcoin Core - {self.max_threads} threads, throttled)"

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'session'):
            self.session.close()

    def process_transaction_batch(self, txids: List[str], start_index: int) -> Tuple[List[Dict], int, int]:
        """Process a batch of transactions in a single thread"""
        bet_txs = []
        runestone_count = 0
        processed_count = 0

        for i, txid in enumerate(txids):
            try:
                # Get transaction details
                tx_details = self.get_transaction_details(txid)
                processed_count += 1

                if not tx_details:
                    continue

                # Check for Runestone and bet transactions
                is_runestone = self.is_runestone_transaction(tx_details)
                is_bet = self.is_bet_transaction(tx_details)

                if is_runestone:
                    runestone_count += 1

                if is_bet:
                    bet_data = {
                        'txid': txid,
                        'position': start_index + i,
                        'fee': tx_details.get('fee', 0),
                        'size': tx_details.get('size', 0),
                        'vsize': tx_details.get('vsize', 0)
                    }
                    bet_txs.append(bet_data)

                    # Log bet transaction
                    self.log_bet_transaction(txid, tx_details, start_index + i)

            except Exception as e:
                self.logger.error(f"Error processing transaction {txid}: {e}")
                continue

        return bet_txs, runestone_count, processed_count

    def get_block_info(self, block_height: int) -> Dict:
        """Get block information including hash and transactions using Bitcoin Core RPC"""
        try:
            # Get block hash from height
            print(f"Getting block hash for height {block_height}...")
            block_hash = self.make_rpc_call("getblockhash", [block_height])

            if not block_hash:
                raise Exception(f"Failed to get block hash for height {block_height}")

            print(f"Block hash: {block_hash}")

            # Get block data with transaction details
            print(f"Getting block data...")
            block_data = self.make_rpc_call("getblock", [block_hash, 1])  # verbosity=1 for tx list

            if not block_data:
                raise Exception(f"Failed to get block data for hash {block_hash}")

            tx_ids = block_data.get('tx', [])
            print(f"Found {len(tx_ids)} transactions in block {block_height}")

            return {
                'height': block_height,
                'hash': block_hash,
                'timestamp': block_data.get('time', 0),
                'tx_count': len(tx_ids),
                'transactions': tx_ids,
                'size': block_data.get('size', 0),
                'weight': block_data.get('weight', 0),
                'difficulty': block_data.get('difficulty', 0)
            }
        except Exception as e:
            print(f"Error getting block info for height {block_height}: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def get_transaction_details(self, txid: str, fetch_prevouts: bool = False) -> Dict:
        """Get detailed transaction information using Bitcoin Core RPC"""
        try:
            # Get raw transaction with verbose output
            tx_data = self.make_rpc_call("getrawtransaction", [txid, True])

            # Check if we got valid data
            if tx_data is None:
                self.logger.warning(f"RPC call returned None for transaction {txid}")
                return {}

            if not isinstance(tx_data, dict):
                self.logger.warning(f"RPC call returned invalid data type for transaction {txid}: {type(tx_data)}")
                return {}

            # Convert to format similar to Blockstream API for compatibility
            formatted_tx = {
                'txid': tx_data.get('txid'),
                'size': tx_data.get('size', 0),
                'vsize': tx_data.get('vsize', 0),
                'weight': tx_data.get('weight', 0),
                'fee': 0,  # Will calculate if needed
                'status': {
                    'confirmed': tx_data.get('confirmations', 0) > 0,
                    'confirmations': tx_data.get('confirmations', 0)
                },
                'vin': [],
                'vout': []
            }

            # Process inputs and optionally fetch previous outputs for sender addresses
            for vin in tx_data.get('vin', []):
                input_data = {
                    'txid': vin.get('txid'),
                    'vout': vin.get('vout'),
                    'prevout': {}
                }

                # Fetch previous transaction output details if requested and not coinbase
                if fetch_prevouts and vin.get('txid') and vin.get('vout') is not None:
                    try:
                        prev_tx = self.make_rpc_call("getrawtransaction", [vin['txid'], True])
                        if prev_tx and 'vout' in prev_tx:
                            prev_vout = prev_tx['vout'][vin['vout']]
                            input_data['prevout'] = {
                                'value': int(prev_vout.get('value', 0) * 100000000),  # Convert to satoshis
                                'scriptpubkey': prev_vout.get('scriptPubKey', {}).get('hex', ''),
                                'scriptpubkey_address': prev_vout.get('scriptPubKey', {}).get('address', '')
                            }
                    except Exception as e:
                        # Don't fail the whole transaction if we can't get prevout
                        self.logger.debug(f"Could not fetch prevout for {vin.get('txid')}:{vin.get('vout')}: {e}")

                formatted_tx['vin'].append(input_data)

            # Process outputs
            for vout in tx_data.get('vout', []):
                output_data = {
                    'value': int(vout.get('value', 0) * 100000000),  # Convert to satoshis
                    'scriptpubkey': vout.get('scriptPubKey', {}).get('hex', ''),
                    'scriptpubkey_address': vout.get('scriptPubKey', {}).get('address', '')
                }
                formatted_tx['vout'].append(output_data)

            return formatted_tx

        except Exception as e:
            self.logger.error(f"Exception getting transaction {txid}: {e}")
            return {}

    def decode_runestone_data(self, script_hex: str) -> Dict:
        """Decode Runestone OP_RETURN data"""
        try:
            if not script_hex.startswith('6a'):
                return {}

            # Extract data after OP_RETURN
            data_hex = script_hex[4:]  # Skip OP_RETURN (6a) and length

            # Handle different length encodings
            if data_hex.startswith('4c'):  # OP_PUSHDATA1
                length = int(data_hex[2:4], 16)
                actual_data = data_hex[4:4+(length*2)]
            elif data_hex.startswith('4d'):  # OP_PUSHDATA2
                length = int(data_hex[6:8] + data_hex[4:6], 16)  # Little endian
                actual_data = data_hex[8:8+(length*2)]
            else:
                # Direct length byte
                length = int(data_hex[:2], 16)
                actual_data = data_hex[2:2+(length*2)]

            return {
                'raw_data': actual_data,
                'length': length,
                'data_bytes': bytes.fromhex(actual_data) if actual_data else b''
            }
        except Exception:
            return {}

    def analyze_tacoclicker_pattern(self, raw_data: str) -> Dict:
        """Analyze TACOCLICKER bet pattern structure"""
        analysis = {
            'is_bet': False,
            'confidence': 0,
            'patterns_found': [],
            'structure': {}
        }

        if not raw_data or len(raw_data) < 20:
            return analysis

        # Core TACOCLICKER patterns
        patterns = {
            'bet_prefix': 'ff7f8196ec82d08bc0a882',  # Always at start
            'bet_marker': 'efa68ada',                # Always present in bets
            'bet_suffix_1': 'ff7f',                  # Common suffix pattern
            'bet_suffix_2': 'ecaebea040',            # Another suffix pattern
            'alkane_marker': 'f9efa68ada',           # Extended alkane pattern
        }

        confidence = 0
        found_patterns = []

        # Check required patterns
        if raw_data.startswith(patterns['bet_prefix']):
            confidence += 40
            found_patterns.append('bet_prefix')

        if patterns['bet_marker'] in raw_data:
            confidence += 30
            found_patterns.append('bet_marker')

        # Check optional patterns that increase confidence
        if patterns['bet_suffix_1'] in raw_data:
            confidence += 10
            found_patterns.append('bet_suffix_1')

        if raw_data.endswith(patterns['bet_suffix_2']):
            confidence += 15
            found_patterns.append('bet_suffix_2')

        if patterns['alkane_marker'] in raw_data:
            confidence += 20
            found_patterns.append('alkane_marker')

        # Additional structural checks
        if len(raw_data) >= 70:  # At least 35 bytes
            confidence += 5

        # Check for hex pattern consistency (all valid hex)
        try:
            bytes.fromhex(raw_data)
            confidence += 5
        except ValueError:
            confidence -= 20

        analysis['confidence'] = confidence
        analysis['patterns_found'] = found_patterns
        analysis['is_bet'] = confidence >= 70  # Require high confidence

        return analysis

    def is_bet_transaction(self, tx_data: Dict) -> bool:
        """Check if transaction is specifically a salsa bet using robust pattern analysis"""
        for output in tx_data.get('vout', []):
            script_hex = output.get('scriptpubkey', '')
            if script_hex.startswith('6a'):
                runestone_data = self.decode_runestone_data(script_hex)
                if runestone_data and runestone_data.get('raw_data'):
                    raw_data = runestone_data['raw_data']

                    # Use comprehensive pattern analysis
                    analysis = self.analyze_tacoclicker_pattern(raw_data)

                    if analysis['is_bet']:
                        return True
        return False

    def is_runestone_transaction(self, tx_data: Dict) -> bool:
        """Check if transaction contains any Runestone OP_RETURN (for general detection)"""
        for output in tx_data.get('vout', []):
            script_hex = output.get('scriptpubkey', '')
            if script_hex.startswith('6a'):
                runestone_data = self.decode_runestone_data(script_hex)
                if runestone_data and runestone_data.get('data_bytes'):
                    # Basic Runestone detection (any OP_RETURN with sufficient data)
                    return len(runestone_data['data_bytes']) >= 10
        return False

    def decode_bet_details(self, tx_data: Dict) -> Dict:
        """Decode bet transaction details using robust pattern analysis"""
        for output in tx_data.get('vout', []):
            script_hex = output.get('scriptpubkey', '')
            if script_hex.startswith('6a'):
                runestone_data = self.decode_runestone_data(script_hex)
                if runestone_data and runestone_data.get('raw_data'):
                    raw_data = runestone_data['raw_data']

                    # Use comprehensive pattern analysis
                    analysis = self.analyze_tacoclicker_pattern(raw_data)

                    if analysis['is_bet']:
                        # Find the position of the middle pattern for variable extraction
                        bet_pattern_middle = 'efa68ada'
                        middle_pos = raw_data.find(bet_pattern_middle)

                        if middle_pos > 0:
                            # Extract variable parts
                            variable_part1 = raw_data[20:middle_pos]  # Between start and middle pattern
                            variable_part2 = raw_data[middle_pos+8:]  # After middle pattern
                        else:
                            variable_part1 = ""
                            variable_part2 = ""

                        return {
                            'is_bet': True,
                            'raw_data': raw_data,
                            'confidence': analysis['confidence'],
                            'patterns_found': analysis['patterns_found'],
                            'variable_part1': variable_part1,
                            'variable_part2': variable_part2,
                            'full_length': len(raw_data) // 2
                        }
        return {'is_bet': False}

    def create_telegram_salsa_summary(self, block_height: int, results: Dict) -> str:
        """Create mobile-friendly Telegram summary using arrow format with fixed mempool link"""
        # The salsa analyzer returns 'candidates', not 'bets'
        candidates = results.get('candidates', [])
        if not candidates:
            return f"Salsa Block {block_height}: No TACOCLICKER bets found"

        bets = candidates  # Use candidates as bets
        winner = results.get('winner')

        # Clean header format inspired by the preferred style
        message_parts = [
            f"🌶️ SALSA BLOCK {block_height} RESULTS",
            f"🎯 {len(bets)} TACOCLICKER bets found",
            ""  # Empty line for spacing
        ]

        if winner:
            # Get winner address (truncated for mobile)
            winner_addr = self.get_sender_address_from_txid(winner['txid'])
            winner_short = winner_addr[:8] + "..." + winner_addr[-6:] if len(winner_addr) > 20 else winner_addr

            message_parts.append(f"🏆 WINNER: {winner_short}")
            # Fix mempool link - use full transaction ID, not truncated
            message_parts.append(f"🔗 mempool.space/tx/{winner['txid']}")
            message_parts.append("")  # Empty line

        # Add "TOP 10 HASHES:" header like in the preferred format
        message_parts.append("📊 TOP 10 HASHES:")

        # Top 10 with arrow format
        for i, candidate in enumerate(bets[:10]):
            # Get sender address (truncated)
            addr = self.get_sender_address_from_txid(candidate['txid'])
            addr_short = addr[:8] + "..." + addr[-6:] if len(addr) > 20 else addr

            # Get hash (truncated)
            hash_full = candidate.get('candidate_hash', '')
            hash_short = hash_full[:8] + "..." + hash_full[-4:] if len(hash_full) > 16 else hash_full

            # Emoji indicators for top 10 (fix emoji 10 issue)
            if i == 0:
                emoji = "🥇"
            elif i == 1:
                emoji = "🥈"
            elif i == 2:
                emoji = "🥉"
            elif i <= 8:  # 4-9 use number emojis
                emoji = f"{i+1}️⃣"
            elif i == 9:  # Position 10 - use different approach
                emoji = "🔟"
            else:
                emoji = f"{i+1:2d}"

            # Arrow format: emoji + address → hash
            message_parts.append(f"{emoji} {addr_short} → {hash_short}")

        return "\n".join(message_parts)





    def get_sender_address_from_txid(self, txid: str) -> str:
        """Get sender address from transaction ID"""
        try:
            tx_data = self.make_rpc_call("getrawtransaction", [txid, True])
            if tx_data and tx_data.get('vin'):
                first_input = tx_data['vin'][0]
                if first_input.get('txid') and first_input.get('vout') is not None:
                    prev_tx = self.make_rpc_call("getrawtransaction", [first_input['txid'], True])
                    if prev_tx and 'vout' in prev_tx:
                        prev_vout = prev_tx['vout'][first_input['vout']]
                        return prev_vout.get('scriptPubKey', {}).get('address', 'Unknown')
        except Exception as e:
            self.logger.debug(f"Error getting sender address for {txid}: {e}")
        return "Unknown"

    def send_telegram_notification(self, analysis_result: Dict):
        """Send salsa block analysis results to Telegram"""
        try:
            # Try to load Telegram config and send notification
            from simple_telegram_bot import load_telegram_config, SimpleTelegramIntegration

            # Load Telegram configuration
            config = load_telegram_config()

            # Create bot integration
            bot = SimpleTelegramIntegration(config['bot_token'])

            # Generate the Telegram summary using the preferred format (arrow format with truncated addresses)
            block_height = analysis_result['block_height']
            telegram_summary = self.create_telegram_salsa_summary(block_height, analysis_result)

            # Send the summary
            success = bot.send_salsa_block_results(telegram_summary)

            if success:
                print(f"Telegram notification sent for block {block_height}")
                self.logger.info(f"Telegram notification sent successfully for block {block_height}")
            else:
                print(f"Telegram notification failed for block {block_height}")
                self.logger.warning(f"Telegram notification failed for block {block_height}")

        except FileNotFoundError:
            print("Telegram config not found - skipping notification")
            self.logger.info("Telegram config not found - skipping notification")
        except ImportError:
            print("Telegram bot module not available - skipping notification")
            self.logger.info("Telegram bot module not available - skipping notification")
        except Exception as e:
            print(f"Telegram notification error: {e}")
            self.logger.error(f"Telegram notification error: {e}")

    def log_bet_transaction(self, txid: str, tx_details: Dict, position: int):
        """Log detailed information about a bet transaction"""
        try:
            # Get sender address - fetch prevouts if not already available
            sender_address = "Unknown"
            if tx_details.get('vin'):
                first_input = tx_details['vin'][0]
                if 'prevout' in first_input and 'scriptpubkey_address' in first_input['prevout']:
                    sender_address = first_input['prevout']['scriptpubkey_address']
                elif first_input.get('txid') and first_input.get('vout') is not None:
                    # Fetch the previous transaction to get sender address
                    try:
                        prev_tx = self.make_rpc_call("getrawtransaction", [first_input['txid'], True])
                        if prev_tx and 'vout' in prev_tx:
                            prev_vout = prev_tx['vout'][first_input['vout']]
                            sender_address = prev_vout.get('scriptPubKey', {}).get('address', 'Unknown')
                    except Exception as e:
                        self.logger.debug(f"Could not fetch sender address for {txid}: {e}")

            # Get OP_RETURN data
            op_return_data = "Not found"
            for output in tx_details.get('vout', []):
                script_hex = output.get('scriptpubkey', '')
                if script_hex.startswith('6a'):
                    runestone_data = self.decode_runestone_data(script_hex)
                    if runestone_data.get('raw_data'):
                        op_return_data = runestone_data['raw_data']
                    break

            # Log comprehensive bet information
            self.logger.info(f"BET TRANSACTION FOUND:")
            self.logger.info(f"  TXID: {txid}")
            self.logger.info(f"  Position in block: {position}")
            self.logger.info(f"  Sender: {sender_address}")
            self.logger.info(f"  Fee: {tx_details.get('fee', 0)} sats")
            self.logger.info(f"  Size: {tx_details.get('size', 0)} bytes")
            self.logger.info(f"  VSize: {tx_details.get('vsize', 0)} bytes")
            self.logger.info(f"  Confirmations: {tx_details.get('status', {}).get('confirmed', False)}")
            self.logger.info(f"  OP_RETURN data: {op_return_data}")

            # Log input/output details
            self.logger.info(f"  Inputs: {len(tx_details.get('vin', []))}")
            for i, inp in enumerate(tx_details.get('vin', [])):
                self.logger.info(f"    Input {i}: {inp.get('txid', 'unknown')}:{inp.get('vout', 'unknown')} - {inp.get('prevout', {}).get('value', 0)} sats")

            self.logger.info(f"  Outputs: {len(tx_details.get('vout', []))}")
            for i, out in enumerate(tx_details.get('vout', [])):
                self.logger.info(f"    Output {i}: {out.get('value', 0)} sats - {out.get('scriptpubkey_address', 'unknown')}")

            self.logger.info("-" * 80)

        except Exception as e:
            self.logger.error(f"Error logging bet transaction {txid}: {e}")

    def calculate_candidate_hash(self, txid: str, block_hash: str) -> Tuple[str, int]:
        """Calculate SHA256(txid XOR blockhash) for salsa competition"""
        try:
            # Ensure both are proper hex strings
            if len(txid) != 64:
                raise ValueError(f"TXID must be 64 hex chars, got {len(txid)}: {txid}")
            if len(block_hash) != 64:
                raise ValueError(f"Block hash must be 64 hex chars, got {len(block_hash)}: {block_hash}")

            # Convert txid to bytes (reverse for little endian)
            txid_bytes = bytes.fromhex(txid)[::-1]

            # Convert block hash to bytes (reverse for little endian)
            block_hash_bytes = bytes.fromhex(block_hash)[::-1]

            # XOR the two hashes
            xor_result = bytes([a ^ b for a, b in zip(txid_bytes, block_hash_bytes)])

            # Calculate SHA256 of XOR result
            candidate_hash = hashlib.sha256(xor_result).digest()

            return candidate_hash.hex(), int.from_bytes(candidate_hash, 'big')
        except Exception as e:
            print(f"Error in calculate_candidate_hash: {e}")
            print(f"TXID: {txid} (len: {len(txid)})")
            print(f"Block hash: {block_hash} (len: {len(block_hash)})")
            raise

    def analyze_salsa_block(self, block_height: int, send_telegram: bool = True) -> Dict:
        """Comprehensive analysis of a salsa block"""
        analysis_start = datetime.now()

        print(f"=== SALSA BLOCK ANALYSIS (Runestone) ===")
        print(f"Block Height: {block_height}")
        print(f"Analysis Time: {analysis_start.strftime('%Y-%m-%d-%H:%M:%S.%f')[:-3]}Z")
        print()

        # Log analysis start
        self.logger.info(f"Starting analysis of block {block_height}")
        self.logger.info(f"Analysis timestamp: {analysis_start.isoformat()}")

        # Get block information
        block_info = self.get_block_info(block_height)
        if not block_info:
            error_msg = "Failed to get block information"
            print(error_msg)
            self.logger.error(error_msg)
            return {}

        block_hash = block_info['hash']
        transactions = block_info['transactions']

        print(f"Block Hash: {block_hash}")
        print(f"Block Hash (bytes): {bytes.fromhex(block_hash)[::-1].hex()}")
        print(f"Total Transactions: {len(transactions)}")
        print()

        # Log block information
        self.logger.info(f"Block hash: {block_hash}")
        self.logger.info(f"Total transactions: {len(transactions)}")
        self.logger.info(f"Block timestamp: {block_info.get('timestamp', 'unknown')}")

        # Scan for actual bet transactions (not just any Runestone)
        print("Scanning for salsa bet transactions...")
        bet_txs = []
        bet_count = 0
        total_runestones = 0

        # Process all transactions using multi-threading for maximum speed!
        total_tx_count = len(transactions)
        processed_count = 0

        print(f"Processing {total_tx_count} transactions with {self.max_threads} threads...")
        self.logger.info(f"Starting multi-threaded transaction processing: {total_tx_count} transactions with {self.max_threads} threads")

        # Split transactions into batches for threading
        batch_size = max(1, total_tx_count // (self.max_threads * 4))  # 4 batches per thread for better load balancing
        batches = []

        for i in range(0, total_tx_count, batch_size):
            batch_txids = transactions[i:i + batch_size]
            batches.append((batch_txids, i))

        print(f"Split into {len(batches)} batches of ~{batch_size} transactions each")

        # Process batches concurrently
        all_bet_txs = []
        total_runestones = 0

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.process_transaction_batch, batch_txids, start_idx): (batch_txids, start_idx)
                for batch_txids, start_idx in batches
            }

            # Collect results as they complete
            completed_batches = 0
            for future in as_completed(future_to_batch):
                batch_txids, _ = future_to_batch[future]  # start_idx not needed here
                completed_batches += 1

                try:
                    batch_bet_txs, batch_runestones, batch_processed = future.result()

                    # Merge results
                    all_bet_txs.extend(batch_bet_txs)
                    total_runestones += batch_runestones
                    processed_count += batch_processed

                    # Show progress
                    rpc_stats = self.get_rpc_usage_stats()
                    print(f"Batch {completed_batches}/{len(batches)} complete | Processed: {processed_count}/{total_tx_count} | Found {len(all_bet_txs)} bets, {total_runestones} Runestones | {rpc_stats}")

                except Exception as e:
                    self.logger.error(f"Batch processing failed: {e}")
                    continue

        # Update counters for compatibility with existing code
        bet_txs = all_bet_txs
        bet_count = len(bet_txs)

        print()
        print(f"Found {bet_count} bet transactions out of {total_runestones} Runestone transactions")
        print()

        if not bet_txs:
            print("No bet transactions found in this block")
            return {
                'block_height': block_height,
                'block_hash': block_hash,
                'total_transactions': len(transactions),
                'runestone_transactions': total_runestones,
                'bet_transactions': 0,
                'candidates': [],
                'winner': None
            }

        # Calculate candidate hashes for all bet transactions
        candidates = []
        for tx in bet_txs:
            txid = tx['txid']
            candidate_hash, candidate_int = self.calculate_candidate_hash(txid, block_hash)

            candidates.append({
                'txid': txid,
                'position': tx['position'],
                'candidate_hash': candidate_hash,
                'candidate_int': candidate_int,
                'fee': tx['fee'],
                'size': tx['size']
            })

        # Sort by candidate hash value (lowest wins)
        candidates.sort(key=lambda x: x['candidate_int'])

        # Log all candidates with their rankings
        self.logger.info("=== CANDIDATE RANKINGS ===")
        for i, candidate in enumerate(candidates, 1):
            self.logger.info(f"Rank {i:2d}: {candidate['txid']} - Hash: {candidate['candidate_hash']} - Fee: {candidate['fee']} sats")

        # Log detailed bet transactions to separate file
        self.bet_logger.info(f"\n=== BLOCK {block_height} BET TRANSACTIONS ===")
        self.bet_logger.info(f"Analysis Time: {datetime.now().isoformat()}")
        self.bet_logger.info(f"Block Hash: {block_hash}")
        self.bet_logger.info(f"Total Bet Transactions: {len(candidates)}")
        self.bet_logger.info("-" * 120)

        for i, candidate in enumerate(candidates, 1):
            # Get sender address
            winner_tx = self.get_transaction_details(candidate['txid'], fetch_prevouts=True)
            sender_address = "Unknown"
            if winner_tx and winner_tx.get('vin'):
                first_input = winner_tx['vin'][0]
                if 'prevout' in first_input and 'scriptpubkey_address' in first_input['prevout']:
                    sender_address = first_input['prevout']['scriptpubkey_address']

            # Log to bet transactions file
            self.bet_logger.info(f"{i:2d} | {candidate['txid']} | Pos:{candidate['position']:4d} | Fee:{candidate['fee']:6d} | Size:{candidate['size']:3d} | {sender_address[:40]:40s} | {candidate['candidate_hash'][:16]}...")

        # Display winner
        if candidates:
            winner = candidates[0]
            print("=== SALSA WINNER ===")
            print(f"Transaction ID: {winner['txid']}")

            # Try to get sender address from transaction details
            winner_tx = self.get_transaction_details(winner['txid'], fetch_prevouts=True)
            sender_address = "Unknown"
            if winner_tx and winner_tx.get('vin'):
                # Try to get address from first input
                first_input = winner_tx['vin'][0]
                if 'prevout' in first_input and 'scriptpubkey_address' in first_input['prevout']:
                    sender_address = first_input['prevout']['scriptpubkey_address']

            print(f"Sender Address: {sender_address}")
            print()
            print("Calculation Details:")

            # Show actual calculation bytes
            txid_bytes = bytes.fromhex(winner['txid'])[::-1]
            block_bytes = bytes.fromhex(block_hash)[::-1]
            xor_bytes = bytes([a ^ b for a, b in zip(txid_bytes, block_bytes)])

            print(f"  TXID (bytes):      {txid_bytes.hex()}")
            print(f"  Block Hash (bytes): {block_bytes.hex()}")
            print(f"  XOR Result:        {xor_bytes.hex()}")
            print(f"  SHA256 Hash:       {winner['candidate_hash']}")
            print()
            print("Bet Parameters:")
            print(f"  Fee: {winner['fee']} sats")
            print(f"  Size: {winner['size']} bytes")
            print()
            print(f"View on Mempool: https://mempool.space/tx/{winner['txid']}")
            print()

            # Log winner details
            self.logger.info("=== SALSA WINNER ===")
            self.logger.info(f"Winner TXID: {winner['txid']}")
            self.logger.info(f"Winner Address: {sender_address}")
            self.logger.info(f"Winning Hash: {winner['candidate_hash']}")
            self.logger.info(f"Position in block: {winner['position']}")
            self.logger.info(f"Fee: {winner['fee']} sats")
            self.logger.info(f"Size: {winner['size']} bytes")

            # Log to bet file
            self.bet_logger.info(f"\n*** WINNER: {winner['txid']}")
            self.bet_logger.info(f"Address: {sender_address}")
            self.bet_logger.info(f"Hash: {winner['candidate_hash']}")
            self.bet_logger.info(f"Fee: {winner['fee']} sats")

        # Display top 10 candidates
        print("=== TOP 10 CANDIDATES ===")

        for i, candidate in enumerate(candidates[:10], 1):
            txid_short = candidate['txid'][:16] + "..."

            # Try to get sender address
            tx_details = self.get_transaction_details(candidate['txid'], fetch_prevouts=True)
            sender_address = "Unknown"
            if tx_details and tx_details.get('vin'):
                first_input = tx_details['vin'][0]
                if 'prevout' in first_input and 'scriptpubkey_address' in first_input['prevout']:
                    sender_address = first_input['prevout']['scriptpubkey_address']
                    # Truncate long addresses
                    if len(sender_address) > 40:
                        sender_address = sender_address[:37] + "..."

            hash_short = candidate['candidate_hash'][:16] + "..."

            print(f"{i:2d}. {txid_short} ({sender_address})")
            print(f"    Hash: {hash_short}")

        analysis_result = {
            'block_height': block_height,
            'block_hash': block_hash,
            'total_transactions': len(transactions),
            'runestone_transactions': total_runestones,
            'bet_transactions': len(bet_txs),
            'candidates': candidates,
            'winner': candidates[0] if candidates else None,
            'analysis_time': datetime.now().isoformat()
        }

        # Log analysis completion
        analysis_end = datetime.now()
        analysis_duration = analysis_end - analysis_start

        # Final RPC usage summary
        final_rpc_stats = self.get_rpc_usage_stats()
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"Analysis duration: {analysis_duration.total_seconds():.2f} seconds")
        print(f"Final {final_rpc_stats}")

        self.logger.info("=== ANALYSIS COMPLETE ===")
        self.logger.info(f"Analysis duration: {analysis_duration.total_seconds():.2f} seconds")
        self.logger.info(f"Total transactions processed: {len(transactions)}")
        self.logger.info(f"Runestone transactions found: {total_runestones}")
        self.logger.info(f"Bet transactions found: {len(bet_txs)}")
        self.logger.info(f"Winner: {candidates[0]['txid'] if candidates else 'None'}")
        self.logger.info(f"Final {final_rpc_stats}")

        # Save results to file
        filename = f"salsa_analysis_block_{block_height}.json"
        try:
            with open(filename, 'w') as f:
                # Create a serializable version (convert candidate_int to string for JSON)
                serializable_result = analysis_result.copy()
                if serializable_result['candidates']:
                    for candidate in serializable_result['candidates']:
                        candidate['candidate_int'] = str(candidate['candidate_int'])
                json.dump(serializable_result, f, indent=2)
            print(f"Analysis results saved to {filename}")
            self.logger.info(f"Results saved to {filename}")
        except Exception as e:
            error_msg = f"Warning: Could not save results to file: {e}"
            print(error_msg)
            self.logger.error(error_msg)

        # Send Telegram notification if enabled
        if send_telegram:
            try:
                self.send_telegram_notification(analysis_result)
            except Exception as e:
                print(f"Warning: Could not send Telegram notification: {e}")
                self.logger.error(f"Telegram notification failed: {e}")

        # Close logging handlers
        self.logger.info("=== LOGGING SESSION COMPLETE ===")

        return analysis_result

def get_salsa_block_info() -> Dict:
    """Get information about current and recent salsa blocks using Bitcoin Core RPC"""
    try:
        # Create a temporary analyzer instance to use RPC
        temp_analyzer = SalsaBlockAnalyzer()
        current_height = temp_analyzer.make_rpc_call("getblockcount")

        # Salsa blocks occur every 144 blocks starting from 908352
        base_salsa = 908352
        blocks_since_base = current_height - base_salsa
        blocks_to_next = 144 - (blocks_since_base % 144)

        if blocks_to_next == 144:
            blocks_to_next = 144  # Current block is a salsa block, next is 144 blocks later

        next_salsa = current_height + blocks_to_next

        # Find the most recent salsa block
        if blocks_to_next == 0:
            most_recent_salsa = current_height
        else:
            most_recent_salsa = current_height - (blocks_since_base % 144)

        print(f"Current block height: {current_height}")
        print(f"Most recent salsa block: {most_recent_salsa}")
        print(f"Next salsa block: {next_salsa} (in {blocks_to_next} blocks)")

        return {
            'current_height': current_height,
            'most_recent_salsa': most_recent_salsa,
            'next_salsa': next_salsa,
            'blocks_to_next': blocks_to_next
        }
    except Exception as e:
        print(f"Error calculating salsa block info: {e}")
        return {'most_recent_salsa': 908352}

def get_salsa_block_info_with_analyzer(analyzer: SalsaBlockAnalyzer) -> Dict:
    """Get information about current and recent salsa blocks using existing analyzer"""
    try:
        current_height = analyzer.make_rpc_call("getblockcount")

        # Salsa blocks occur every 144 blocks starting from 908352
        base_salsa = 908352
        blocks_since_base = current_height - base_salsa
        blocks_to_next = 144 - (blocks_since_base % 144)

        if blocks_to_next == 144:
            blocks_to_next = 144  # Current block is a salsa block, next is 144 blocks later

        next_salsa = current_height + blocks_to_next

        # Find the most recent salsa block
        if blocks_to_next == 144:
            most_recent_salsa = current_height
        else:
            most_recent_salsa = current_height - (blocks_since_base % 144)

        print(f"Current block height: {current_height}")
        print(f"Most recent salsa block: {most_recent_salsa}")
        print(f"Next salsa block: {next_salsa} (in {blocks_to_next} blocks)")

        return {
            'current_height': current_height,
            'most_recent_salsa': most_recent_salsa,
            'next_salsa': next_salsa,
            'blocks_to_next': blocks_to_next
        }
    except Exception as e:
        print(f"Error calculating salsa block info: {e}")
        return {'most_recent_salsa': 908352}

def analyze_block_range(analyzer: SalsaBlockAnalyzer, start_block: int, end_block: int):
    """Analyze a range of salsa blocks"""
    print(f"\n=== ANALYZING BLOCK RANGE {start_block} to {end_block} ===")

    total_blocks = end_block - start_block + 1
    all_results = []
    total_bets_found = 0
    total_runestones_found = 0

    for i, block_height in enumerate(range(start_block, end_block + 1), 1):
        print(f"\n--- Block {block_height} ({i}/{total_blocks}) ---")

        try:
            analysis = analyzer.analyze_salsa_block(block_height)

            if analysis:
                all_results.append(analysis)
                bet_count = len(analysis.get('candidates', []))
                runestone_count = analysis.get('runestone_transactions', 0)

                total_bets_found += bet_count
                total_runestones_found += runestone_count

                if bet_count > 0:
                    winner = analysis.get('winner')
                    print(f"[OK] Block {block_height}: {bet_count} bets, Winner: {winner['txid'][:16]}..." if winner else f"[OK] Block {block_height}: {bet_count} bets, No winner")
                else:
                    print(f"[INFO] Block {block_height}: {runestone_count} Runestones, 0 bets")
            else:
                print(f"[ERROR] Block {block_height}: Analysis failed")

        except KeyboardInterrupt:
            print(f"\n[STOP] Analysis interrupted at block {block_height}")
            break
        except Exception as e:
            print(f"[ERROR] Error analyzing block {block_height}: {e}")
            continue

    # Summary
    print(f"\n=== RANGE ANALYSIS SUMMARY ===")
    print(f"Blocks analyzed: {len(all_results)}/{total_blocks}")
    print(f"Total bet transactions found: {total_bets_found}")
    print(f"Total Runestone transactions found: {total_runestones_found}")
    print(f"Blocks with bets: {sum(1 for r in all_results if len(r.get('candidates', [])) > 0)}")

    return all_results

def main():
    """Main analysis function with support for single blocks, ranges, and threading"""
    try:
        # Parse command line arguments
        import sys

        # Check for thread count argument
        max_threads = 4  # Conservative default to avoid overwhelming node
        args = sys.argv[1:]

        if args and args[0].startswith('--threads='):
            max_threads = int(args[0].split('=')[1])
            args = args[1:]  # Remove thread argument
            print(f"Using {max_threads} threads (max 6 for stability)")

        analyzer = SalsaBlockAnalyzer(max_threads=max_threads)

        # Get salsa block info first
        salsa_info = get_salsa_block_info_with_analyzer(analyzer)
        print()

        if len(args) > 0:
            if len(args) == 1:
                # Single block
                block_height = int(args[0])
                analysis = analyzer.analyze_salsa_block(block_height)

                if analysis and analysis.get('candidates'):
                    if analysis.get('winner'):
                        print(f"\nAnalysis complete for block {block_height}")
                        print(f"Found {len(analysis['candidates'])} bet transactions")
                    else:
                        print(f"Analysis complete for block {block_height} - found {len(analysis['candidates'])} bet transactions but no clear winner")
                else:
                    print(f"Analysis complete for block {block_height} - no bet transactions found")

            elif len(args) == 2:
                # Block range
                start_block = int(args[0])
                end_block = int(args[1])

                if start_block > end_block:
                    print("Error: Start block must be less than or equal to end block")
                    return

                analyze_block_range(analyzer, start_block, end_block)

            else:
                print("Usage:")
                print("  python salsa_block_analyzer.py [--threads=N] [block_height]")
                print("  python salsa_block_analyzer.py [--threads=N] [start_block] [end_block]")
                print("  --threads=N: Use N threads (default: 4, max: 6 for stability)")
                return

        else:
            # Use the most recent salsa block for real data
            block_height = salsa_info.get('most_recent_salsa', 908352)
            print(f"No block height specified, analyzing most recent salsa block: {block_height}")
            print()

            analysis = analyzer.analyze_salsa_block(block_height)

            if analysis and analysis.get('candidates'):
                if analysis.get('winner'):
                    print(f"\nAnalysis complete for block {block_height}")
                    print(f"Found {len(analysis['candidates'])} bet transactions")
                else:
                    print(f"Analysis complete for block {block_height} - found {len(analysis['candidates'])} bet transactions but no clear winner")
            else:
                print(f"Analysis complete for block {block_height} - no bet transactions found")

    except ValueError as e:
        print(f"Error: Invalid block height(s) or thread count provided: {e}")
        print("Usage:")
        print("  python salsa_block_analyzer.py [--threads=N] [block_height]")
        print("  python salsa_block_analyzer.py [--threads=N] [start_block] [end_block]")
        print("  --threads=N: Use N threads (default: 4, max: 6 for stability)")
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
