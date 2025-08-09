#!/usr/bin/env python3
"""
TACOCLICKER Mempool Monitor - Real-time Bet Detection
Monitors Bitcoin mempool and new blocks for TACOCLICKER bet transactions
Uses Bitcoin Core RPC for maximum performance and real-time detection
"""

import requests
import hashlib
import time
import json
import os
import logging
import threading
from datetime import datetime
from typing import List, Dict, Set, Optional
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


# Import Telegram bot integration
try:
    from simple_telegram_bot import SimpleTelegramIntegration, load_telegram_config
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("Telegram integration not available")

class TacoClickerMempoolMonitor:
    """Real-time monitor for TACOCLICKER bet transactions in mempool and blocks"""

    def __init__(self, rpc_credentials_file: str = "bitcoin_rpc_credentials.json", max_threads: int = 4, enable_telegram: bool = False, simulation_mode: bool = False):
        # Load Bitcoin Core RPC credentials
        self.rpc_credentials = self.load_rpc_credentials(rpc_credentials_file)

        # Initialize Telegram bot if requested (will be set up later with salsa block info)
        self.telegram_bot = None
        self.enable_telegram = enable_telegram
        if enable_telegram and not TELEGRAM_AVAILABLE:
            print("Telegram requested but not available (install python-telegram-bot)")
            self.enable_telegram = False

        # Simulation mode: inject fake bet flows for realistic testing
        self.simulation_mode = simulation_mode
        self.simulated_bets: Set[str] = set()
        self.simulation_start_time = time.time()
        self.last_simulation_event = time.time()

        # RPC connection settings
        self.rpc_url = f"http://{self.rpc_credentials['rpchost']}:{self.rpc_credentials['rpcport']}"
        self.rpc_auth = (self.rpc_credentials['rpcuser'], self.rpc_credentials['rpcpassword'])

        # Thread-safe tracking
        self.rpc_calls_made = 0
        self.rpc_lock = threading.Lock()
        self.max_threads = min(max_threads, 4)  # Conservative for mempool monitoring

        # NEW: Real-time bet tracking with variations
        self.current_mempool_bets: Set[str] = set()  # Current TACOCLICKER bets in next-block candidates
        self.last_bet_count: int = 0  # Previous bet count for change detection
        self.last_block_height: Optional[int] = None
        self.last_mempool_check: float = 0
        self.last_block_time: float = 0.0  # Used to reset mempool baseline after blocks

        # Batching for mempool notifications (reduce chat noise)
        self.batch_window_seconds: float = 10.0
        self._batch_accumulator = {
            'new_bets': set(),
            'removed_bets': set(),
            'last_reported_count': 0,
        }

        # Block comparison tracking
        self.expected_bets_for_next_block: Set[str] = set()  # Bets we expect to be mined
        self.mined_vs_expected_stats: Dict = {
            'total_blocks_analyzed': 0,
            'total_expected_bets': 0,
            'total_actual_bets': 0,
            'accuracy_rate': 0.0
        }

        # Statistics tracking
        self.total_unique_bets_seen: int = 0  # Lifetime unique bets
        self.session_start_time: float = time.time()
        self.last_summary_time: float = time.time()
        self.hourly_new_bets: int = 0
        self.hourly_removed_bets: int = 0

        # Simulation cadence
        self.sim_normal_range = (0, 2)   # bets per batch before normal blocks
        self.sim_salsa_range = (20, 50)  # bets per batch before salsa blocks

        # Setup connection session
        self.session = requests.Session()
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

        # Setup logging
        self.setup_logging()

        # Test connection
        self.test_rpc_connection()

    def load_rpc_credentials(self, credentials_file: str) -> Dict:
        """Load Bitcoin Core RPC credentials from file"""
        possible_paths = [
            Path(credentials_file),
            Path(__file__).parent / credentials_file,
            Path.cwd() / credentials_file,
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
            raise ConnectionError(error_msg)

    def setup_logging(self):
        """Setup logging for mempool monitoring"""
        log_dir = "Log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Main logger
        self.logger = logging.getLogger('MempoolMonitor')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        # Log file handler
        log_file = os.path.join(log_dir, f"mempool_monitor_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Bet transactions log
        bet_log_file = os.path.join(log_dir, f"mempool_bets_{timestamp}.log")
        self.bet_file_handler = logging.FileHandler(bet_log_file, encoding='utf-8')
        self.bet_file_handler.setLevel(logging.INFO)

        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        bet_formatter = logging.Formatter('%(message)s')

        file_handler.setFormatter(detailed_formatter)
        self.bet_file_handler.setFormatter(bet_formatter)

        self.logger.addHandler(file_handler)

        # Separate bet logger
        self.bet_logger = logging.getLogger('MempoolBets')
        self.bet_logger.setLevel(logging.INFO)
        self.bet_logger.handlers.clear()
        self.bet_logger.addHandler(self.bet_file_handler)

        self.logger.info("=== TACOCLICKER MEMPOOL MONITOR STARTED ===")
        self.bet_logger.info("=== MEMPOOL BET TRANSACTIONS LOG ===")
        self.bet_logger.info("Format: Timestamp | Source | TXID | Fee | Size | Sender | OP_RETURN")
        self.bet_logger.info("=" * 120)

    def make_rpc_call(self, method: str, params: List = None) -> Dict:
        """Make a Bitcoin Core RPC call"""
        if params is None:
            params = []

        with self.rpc_lock:
            call_id = self.rpc_calls_made
            self.rpc_calls_made += 1

        payload = {
            "jsonrpc": "1.0",
            "id": f"mempool_monitor_{call_id}",
            "method": method,
            "params": params
        }

        try:
            response = self.session.post(
                self.rpc_url,
                json=payload,
                auth=self.rpc_auth,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
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
            self.logger.error(f"RPC call '{method}' failed: {e}")
            return None

    def get_mempool_txids(self) -> List[str]:
        """Get all transaction IDs currently in mempool"""
        try:
            # Get raw mempool (just txids for speed)
            mempool_txids = self.make_rpc_call("getrawmempool", [False])
            return mempool_txids if mempool_txids else []
        except Exception as e:
            self.logger.error(f"Error getting mempool txids: {e}")
            return []

    def get_next_block_candidates(self) -> List[str]:
        """Get transactions that are likely to be in the next block (high priority).
        In simulation_mode we synthesize candidate txids to emulate load.
        """
        if self.simulation_mode:
            return list(self.simulated_bets)

        try:
            # Get block template - this shows what would be in the next block
            template = self.make_rpc_call("getblocktemplate", [{"rules": ["segwit"]}])

            if template and 'transactions' in template:
                # Extract transaction IDs from block template
                candidate_txids = []
                for tx in template['transactions']:
                    if 'txid' in tx:
                        candidate_txids.append(tx['txid'])

                self.logger.debug(f"Found {len(candidate_txids)} next-block candidates")
                return candidate_txids
            else:
                # Fallback: get mempool with fee info and filter high-fee transactions
                return self.get_high_priority_mempool_txs()

        except Exception as e:
            self.logger.debug(f"Error getting block template: {e}")
            # Fallback to high-priority mempool transactions
            return self.get_high_priority_mempool_txs()

    def get_high_priority_mempool_txs(self) -> List[str]:
        """Fallback: Get high-priority mempool transactions based on fee rate"""
        try:
            # Get mempool with verbose info (includes fee rates)
            mempool_info = self.make_rpc_call("getrawmempool", [True])

            if not mempool_info:
                return []

            # Sort by fee rate (descendingfee per vbyte)
            tx_list = []
            for txid, info in mempool_info.items():
                fee_rate = info.get('fee', 0) / max(info.get('vsize', 1), 1) * 100000000  # sats/vbyte
                tx_list.append((txid, fee_rate))

            # Sort by fee rate descending and take top candidates
            tx_list.sort(key=lambda x: x[1], reverse=True)

            # Take top 2000 transactions (typical block size)
            high_priority_txs = [txid for txid, _ in tx_list[:2000]]

            self.logger.debug(f"Found {len(high_priority_txs)} high-priority mempool transactions")
            return high_priority_txs

        except Exception as e:
            self.logger.error(f"Error getting high-priority mempool transactions: {e}")
            return []

    def get_transaction_details(self, txid: str) -> Dict:
        """Get detailed transaction information.
        In simulation_mode, synthesize a Runestone OP_RETURN-esque output for any txid.
        """
        if self.simulation_mode:
            # Fake but structurally correct shape with an OP_RETURN-like output
            return {
                'txid': txid,
                'size': 200,
                'vsize': 180,
                'weight': 720,
                'fee': 1000,
                'confirmations': 0,
                'vin': [{'txid': '00'*32, 'vout': 0}],
                'vout': [
                    {
                        'value': 0,
                        'scriptpubkey': '6a' + '4c' + '2a' + 'ff7f8196ec82d08bc0a882' + 'efa68ada' + 'abcd'*6 + 'ecaebea040',
                        'scriptpubkey_address': ''
                    }
                ]
            }

        try:
            # Get raw transaction with verbose output
            tx_data = self.make_rpc_call("getrawtransaction", [txid, True])

            if not tx_data:
                return {}

            # Convert to standardized format
            formatted_tx = {
                'txid': tx_data.get('txid'),
                'size': tx_data.get('size', 0),
                'vsize': tx_data.get('vsize', 0),
                'weight': tx_data.get('weight', 0),
                'fee': 0,  # Will calculate if needed
                'confirmations': tx_data.get('confirmations', 0),
                'vin': [],
                'vout': []
            }

            # Process outputs to find OP_RETURN
            for vout in tx_data.get('vout', []):
                output_data = {
                    'value': int(vout.get('value', 0) * 100000000),  # Convert to satoshis
                    'scriptpubkey': vout.get('scriptPubKey', {}).get('hex', ''),
                    'scriptpubkey_address': vout.get('scriptPubKey', {}).get('address', '')
                }
                formatted_tx['vout'].append(output_data)

            return formatted_tx

        except Exception as e:
            self.logger.debug(f"Error getting transaction {txid}: {e}")
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

    def is_tacoclicker_bet(self, tx_data: Dict) -> bool:
        """Check if transaction is a TACOCLICKER bet using robust pattern analysis"""
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

    def get_op_return_data(self, tx_data: Dict) -> str:
        """Extract OP_RETURN data from transaction"""
        for output in tx_data.get('vout', []):
            script_hex = output.get('scriptpubkey', '')
            if script_hex.startswith('6a'):
                runestone_data = self.decode_runestone_data(script_hex)
                if runestone_data.get('raw_data'):
                    return runestone_data['raw_data']
        return "Not found"

    def get_sender_address(self, txid: str) -> str:
        """Get sender address from first input (simplified)"""
        try:
            tx_data = self.make_rpc_call("getrawtransaction", [txid, True])
            if tx_data and tx_data.get('vin'):
                first_input = tx_data['vin'][0]
                if first_input.get('txid') and first_input.get('vout') is not None:
                    prev_tx = self.make_rpc_call("getrawtransaction", [first_input['txid'], True])
                    if prev_tx and 'vout' in prev_tx:
                        prev_vout = prev_tx['vout'][first_input['vout']]
                        return prev_vout.get('scriptPubKey', {}).get('address', 'Unknown')
        except Exception:
            pass
        return "Unknown"

    def log_bet_transaction(self, txid: str, tx_details: Dict, source: str):
        """Log a detected bet transaction"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            sender_address = self.get_sender_address(txid)
            op_return_data = self.get_op_return_data(tx_details)

            # Log to console
            print(f" [{timestamp}] TACOCLICKER BET DETECTED ({source})")
            print(f"   TXID: {txid}")
            print(f"   Sender: {sender_address}")
            print(f"   Fee: {tx_details.get('fee', 0)} sats")
            print(f"   Size: {tx_details.get('size', 0)} bytes")
            print(f"   OP_RETURN: {op_return_data[:32]}...")
            print()

            # Log to file
            self.logger.info(f"BET DETECTED ({source}): {txid} - Sender: {sender_address}")

            # Log to bet file
            self.bet_logger.info(f"{timestamp} | {source:8s} | {txid} | {tx_details.get('fee', 0):6d} | {tx_details.get('size', 0):3d} | {sender_address[:40]:40s} | {op_return_data[:32]}")

        except Exception as e:
            self.logger.error(f"Error logging bet transaction {txid}: {e}")

    def scan_mempool_bet_changes(self) -> Dict:
        """Scan for TACOCLICKER bet count changes in next-block candidates"""
        result = {
            'new_bets': [],
            'removed_bets': [],
            'current_count': 0,
            'previous_count': 0,
            'net_change': 0,
            'total_candidates': 0
        }

        try:
            # Get next-block candidates (high priority transactions)
            next_block_candidates = set(self.get_next_block_candidates())
            result['total_candidates'] = len(next_block_candidates)

            if not next_block_candidates:
                print("No next-block candidates found")
                return result

            # Find TACOCLICKER bets among candidates
            current_bets = set()

            for txid in next_block_candidates:
                try:
                    tx_details = self.get_transaction_details(txid)
                    if tx_details and self.is_tacoclicker_bet(tx_details):
                        current_bets.add(txid)
                except Exception as e:
                    self.logger.debug(f"Error checking candidate {txid}: {e}")
                    continue

            # Calculate changes
            new_bets = current_bets - self.current_mempool_bets
            removed_bets = self.current_mempool_bets - current_bets

            # Batch accumulation to reduce chat noise
            self._batch_accumulator['new_bets'].update(new_bets)
            self._batch_accumulator['removed_bets'].update(removed_bets)

            # Fill result with current instantaneous snapshot (used by UI/logic)
            result['new_bets'] = list(self._batch_accumulator['new_bets'])
            result['removed_bets'] = list(self._batch_accumulator['removed_bets'])
            result['current_count'] = len(current_bets)
            result['previous_count'] = self.last_bet_count
            result['net_change'] = len(current_bets) - self.last_bet_count

            # Log new bets (just once when first seen)
            for txid in new_bets:
                try:
                    tx_details = self.get_transaction_details(txid)
                    if tx_details:
                        self.log_bet_transaction(txid, tx_details, "NEW-BET")
                        self.total_unique_bets_seen += 1
                        self.hourly_new_bets += 1
                except Exception as e:
                    self.logger.debug(f"Error logging new bet {txid}: {e}")

            # Log removed bets (file log only)
            for txid in removed_bets:
                self.logger.info(f"BET REMOVED from next-block candidates (low fee): {txid}")
                self.hourly_removed_bets += 1

            # Update state
            self.current_mempool_bets = current_bets
            self.expected_bets_for_next_block = current_bets.copy()
            self.last_bet_count = len(current_bets)

        except Exception as e:
            self.logger.error(f"Error scanning mempool bet changes: {e}")

        return result

    def analyze_mined_block(self, block_height: int) -> Dict:
        """Analyze mined block and compare with expected bets"""
        result = {
            'actual_bets': [],
            'expected_bets': list(self.expected_bets_for_next_block),
            'correctly_predicted': [],
            'missed_predictions': [],
            'unexpected_bets': [],
            'accuracy_rate': 0.0
        }

        try:
            # Get block hash
            block_hash = self.make_rpc_call("getblockhash", [block_height])
            if not block_hash:
                return result

            # Get block data
            block_data = self.make_rpc_call("getblock", [block_hash, 1])
            if not block_data:
                return result

            tx_ids = block_data.get('tx', [])

            # Find TACOCLICKER bets in the mined block
            actual_bets = []
            for txid in tx_ids:
                try:
                    tx_details = self.get_transaction_details(txid)
                    if tx_details and self.is_tacoclicker_bet(tx_details):
                        actual_bets.append(txid)
                except Exception as e:
                    self.logger.debug(f"Error checking mined transaction {txid}: {e}")
                    continue

            # Compare expected vs actual
            expected_set = set(self.expected_bets_for_next_block)
            actual_set = set(actual_bets)

            correctly_predicted = expected_set & actual_set
            missed_predictions = expected_set - actual_set  # Expected but not mined
            unexpected_bets = actual_set - expected_set     # Mined but not expected

            # Calculate accuracy: correct predictions / max(expected, actual)
            total_expected = len(expected_set)
            total_actual = len(actual_set)
            total_correct = len(correctly_predicted)

            if total_expected == 0 and total_actual == 0:
                accuracy = 100.0  # Perfect if both are zero
            elif total_expected == 0:
                accuracy = 0.0     # We expected nothing but got something
            else:
                # Standard accuracy: correct / expected
                accuracy = (total_correct / total_expected * 100)

            result.update({
                'actual_bets': actual_bets,
                'correctly_predicted': list(correctly_predicted),
                'missed_predictions': list(missed_predictions),
                'unexpected_bets': list(unexpected_bets),
                'accuracy_rate': accuracy
            })

            # Update statistics
            self.mined_vs_expected_stats['total_blocks_analyzed'] += 1
            self.mined_vs_expected_stats['total_expected_bets'] += total_expected
            self.mined_vs_expected_stats['total_actual_bets'] += len(actual_bets)

            # Update running accuracy
            total_blocks = self.mined_vs_expected_stats['total_blocks_analyzed']
            if total_blocks > 0:
                overall_accuracy = (self.mined_vs_expected_stats['total_expected_bets'] /
                                  max(self.mined_vs_expected_stats['total_actual_bets'], 1) * 100)
                self.mined_vs_expected_stats['accuracy_rate'] = overall_accuracy

            # Log results
            print(f"Block {block_height} Analysis:")
            print(f"   Expected: {total_expected} bets")
            print(f"   Actual: {len(actual_bets)} bets")
            print(f"   Correctly predicted: {total_correct} ({accuracy:.1f}%)")

            if missed_predictions:
                print(f"Missed (expected but not mined): {len(missed_predictions)}")
            if unexpected_bets:
                print(f"Unexpected (mined but not expected): {len(unexpected_bets)}")

            # Reset expected bets for next block
            self.expected_bets_for_next_block.clear()

        except Exception as e:
            self.logger.error(f"Error analyzing mined block {block_height}: {e}")

        return result

    def run_salsa_block_analysis(self, block_height: int) -> str:
        """Run salsa block analysis and return Telegram summary"""
        try:
            # Import and run salsa block analyzer
            from salsa_block_analyzer import SalsaBlockAnalyzer

            analyzer = SalsaBlockAnalyzer(
                rpc_credentials_file="bitcoin_rpc_credentials.json"
            )

            # Analyze the salsa block
            # If our monitor has Telegram enabled, we will send the Telegram message ourselves
            # to control formatting and avoid duplicates. Otherwise, allow the analyzer to send.
            results = analyzer.analyze_salsa_block(block_height, send_telegram=not self.enable_telegram)

            if results and results.get('candidates'):
                # Create concise Telegram summary (preferred truncated format)
                telegram_summary = analyzer.create_telegram_salsa_summary(block_height, results)

                self.logger.info(f"Salsa block {block_height} analysis completed: {len(results['candidates'])} bets")
                return telegram_summary
            else:
                return f"Salsa Block {block_height}: No TACOCLICKER bets found"

        except Exception as e:
            self.logger.error(f"Error running salsa block analysis for block {block_height}: {e}")
            return f"Salsa Block {block_height}: Analysis failed"

    def is_salsa_block(self, block_height: int) -> bool:
        """Check if block is a salsa block (every 144 blocks starting from 908352)"""
        base_salsa = 908352
        return (block_height - base_salsa) % 144 == 0 and block_height >= base_salsa

    def get_current_block_height(self) -> Optional[int]:
        """Get current blockchain height"""
        try:
            return self.make_rpc_call("getblockcount")
        except Exception as e:
            self.logger.error(f"Error getting block height: {e}")
            return None

    def get_next_salsa_block(self, current_height: int) -> int:
        """Get the next salsa block height"""
        base_salsa = 908352
        blocks_since_base = current_height - base_salsa
        blocks_to_next = 144 - (blocks_since_base % 144)

        if blocks_to_next == 144:
            blocks_to_next = 144  # Current block is a salsa block, next is 144 blocks later

        return current_height + blocks_to_next

    def setup_telegram_bot(self, next_salsa: int):
        """Set up Telegram bot with next salsa block info"""
        if self.enable_telegram and TELEGRAM_AVAILABLE and not self.telegram_bot:
            try:
                telegram_config = load_telegram_config()
                self.telegram_bot = SimpleTelegramIntegration(
                    telegram_config['bot_token']
                )
                if self.telegram_bot.start_bot(next_salsa):
                    print("Telegram bot integration enabled")
                    return True
                else:
                    print("Failed to start Telegram bot")
                    self.telegram_bot = None
                    return False
            except Exception as e:
                print(f"Telegram bot setup failed: {e}")
                self.telegram_bot = None
                return False
        return False

    def monitor_realtime(self, mempool_interval: int = 10, block_check_interval: int = 5):
        """Main monitoring loop for real-time bet detection"""
        print("=== TACOCLICKER MEMPOOL MONITOR STARTED ===")
        print(f"Mempool scan interval: {mempool_interval} seconds")
        print(f"Block check interval: {block_check_interval} seconds")
        print("Press Ctrl+C to stop")
        print()

        # Initialize with current block height
        self.last_block_height = self.get_current_block_height()
        if self.last_block_height:
            print(f"Starting monitoring at block {self.last_block_height}")

            # Set up Telegram bot with next salsa block info
            if self.enable_telegram:
                next_salsa = self.get_next_salsa_block(self.last_block_height)
                self.setup_telegram_bot(next_salsa)

        self.logger.info(f"Real-time monitoring started at block {self.last_block_height}")

        try:
            while True:
                current_time = time.time()

                # Simulation: inject random flows periodically
                if self.simulation_mode and (current_time - self.last_simulation_event) >= max(3, self.batch_window_seconds/8):
                    try:
                        # Decide next-block context
                        height_hint = self.last_block_height or self.get_current_block_height() or 0
                        is_salsa_soon = ((self.get_next_salsa_block(height_hint) - height_hint) <= 1)
                        low, high = (self.sim_salsa_range if is_salsa_soon else self.sim_normal_range)

                        import random, secrets
                        add = random.randint(low, high)
                        # Optionally remove some existing after initial adds
                        remove = random.randint(0, max(0, add // 3)) if self.simulated_bets else 0

                        # Generate new fake txids
                        for _ in range(add):
                            txid = secrets.token_hex(32)
                            self.simulated_bets.add(txid)
                        # Remove a few to simulate falling behind
                        for txid in list(self.simulated_bets)[:remove]:
                            self.simulated_bets.discard(txid)

                        self.last_simulation_event = current_time
                    except Exception:
                        pass

                # Check for new blocks
                current_height = self.get_current_block_height()
                if current_height and current_height != self.last_block_height:
                    if self.last_block_height is not None:
                        # New block detected
                        for height in range(self.last_block_height + 1, current_height + 1):
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            is_salsa = self.is_salsa_block(height)
                            salsa_indicator = " -SALSA BLOCK!" if is_salsa else ""
                            print(f"[{timestamp}] New Block {height}{salsa_indicator}")

                            if is_salsa:
                                # SALSA BLOCK - Run full salsa analysis
                                print(f"SALSA BLOCK {height} - Running full analysis...")

                                # Send initial salsa block notification
                                if self.telegram_bot:
                                    try:
                                        self.telegram_bot.send_salsa_block_alert(height)
                                        print(f"Sent salsa block mined notification for block {height}")
                                    except Exception as e:
                                        print(f"Telegram salsa block notification failed: {e}")
                                        self.logger.error(f"Telegram salsa block notification failed: {e}")

                                # Run comprehensive salsa block analysis
                                try:
                                    salsa_results = self.run_salsa_block_analysis(height)
                                    if salsa_results and self.telegram_bot:
                                        self.telegram_bot.send_salsa_block_results(salsa_results)
                                        print(f"Sent salsa block analysis results for block {height}")
                                    elif not salsa_results:
                                        print(f"No salsa analysis results generated for block {height}")
                                    elif not self.telegram_bot:
                                        print(f"No Telegram bot available for salsa results")
                                except Exception as e:
                                    print(f"Salsa block analysis failed: {e}")
                                    self.logger.error(f"Salsa block analysis failed: {e}")

                                # After a salsa block is mined, refresh mempool baseline and batch accumulators
                                self.current_mempool_bets.clear()
                                self.expected_bets_for_next_block.clear()
                                self.last_bet_count = 0
                                self._batch_accumulator['new_bets'].clear()
                                self._batch_accumulator['removed_bets'].clear()
                                self._batch_accumulator['last_reported_count'] = 0
                                self.last_mempool_check = time.time()

                            else:
                                # NORMAL BLOCK - Only run expected vs actual analysis
                                block_analysis = self.analyze_mined_block(height)

                                # Send Telegram block analysis for normal blocks (expected vs actual)
                                if self.telegram_bot and (block_analysis['actual_bets'] or block_analysis['expected_bets']):
                                    try:
                                        self.telegram_bot.send_block_analysis(
                                            height,
                                            len(block_analysis['expected_bets']),
                                            len(block_analysis['actual_bets']),
                                            block_analysis['accuracy_rate']
                                        )
                                    except Exception as e:
                                        self.logger.debug(f"Telegram block analysis failed: {e}")

                                # After a block is mined, refresh mempool baseline and batch accumulators
                                self.current_mempool_bets.clear()
                                self.expected_bets_for_next_block.clear()
                                self.last_bet_count = 0
                                self._batch_accumulator['new_bets'].clear()
                                self._batch_accumulator['removed_bets'].clear()
                                self._batch_accumulator['last_reported_count'] = 0
                                self.last_mempool_check = time.time()

                    self.last_block_height = current_height

                # Check mempool bet changes periodically
                if current_time - self.last_mempool_check >= mempool_interval:
                    bet_changes = self.scan_mempool_bet_changes()

                    # Check if batch window has elapsed for Telegram notifications
                    time_since_last_batch = current_time - self.last_mempool_check
                    batch_window_elapsed = time_since_last_batch >= self.batch_window_seconds

                    # Show changes to console immediately for observability
                    if bet_changes['net_change'] != 0 or bet_changes['new_bets'] or bet_changes['removed_bets']:
                        current_count = bet_changes['current_count']
                        previous_count = bet_changes['previous_count']
                        new_count = len(bet_changes['new_bets'])
                        removed_count = len(bet_changes['removed_bets'])

                        print(f"TACOCLICKER Bet Count: {current_count} (was {previous_count})")
                        if new_count > 0:
                            print(f"{new_count} new bets added to next-block candidates")
                        if removed_count > 0:
                            print(f"{removed_count} bets removed (low fees)")

                    # Send Telegram notification if batch window elapsed and we have accumulated changes
                    if self.telegram_bot and batch_window_elapsed:
                        accumulated_new = len(self._batch_accumulator['new_bets'])
                        accumulated_removed = len(self._batch_accumulator['removed_bets'])

                        if accumulated_new > 0 or accumulated_removed > 0:
                            try:
                                next_salsa = self.get_next_salsa_block(current_height) if current_height else 0
                                # Calculate previous count for the batch
                                batch_previous_count = bet_changes['current_count'] - accumulated_new + accumulated_removed

                                self.telegram_bot.send_bet_count_update(
                                    bet_changes['current_count'], batch_previous_count,
                                    accumulated_new, accumulated_removed,
                                    next_salsa, current_height
                                )
                                print(f"Sent Telegram batch update: {bet_changes['current_count']} bets (+{accumulated_new} new, -{accumulated_removed} removed)")
                            except Exception as e:
                                self.logger.debug(f"Telegram bet count update failed: {e}")
                                print(f"Telegram bet update failed: {e}")

                        # Reset batch window timer and clear accumulators
                        self._batch_accumulator['new_bets'].clear()
                        self._batch_accumulator['removed_bets'].clear()
                        self._batch_accumulator['last_reported_count'] = bet_changes['current_count']
                        self.last_mempool_check = current_time

                # Show periodic summary (every minute)
                if current_time - self.last_summary_time >= 60:  # 1 minute
                    self.show_summary()
                    self.last_summary_time = current_time

                # Sleep for block check interval
                time.sleep(block_check_interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            self.show_final_summary()
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            print(f"Monitoring error: {e}")
            self.show_final_summary()
            self.logger.error(f"Monitoring error: {e}")

    def show_final_summary(self):
        """Show final summary when monitoring stops"""
        current_time = time.time()
        session_duration = current_time - self.session_start_time
        hours_running = session_duration / 3600
        total_bets = self.total_mempool_bets + self.total_block_bets

        print(f"\n=== FINAL SESSION SUMMARY ===")
        print(f"   Total Duration: {hours_running:.2f} hours")
        print(f"   Total Bets Detected: {total_bets}")
        print(f"      From Mempool: {self.total_mempool_bets}")
        print(f"      From Blocks: {self.total_block_bets}")

        if hours_running > 0:
            rate_per_hour = total_bets / hours_running
            print(f"   Average Rate: {rate_per_hour:.1f} bets/hour")

        if session_duration > 60:
            minutes = session_duration / 60
            rate_per_minute = total_bets / minutes
            print(f"   Average Rate: {rate_per_minute:.2f} bets/minute")

        print(f"   Total RPC Calls: {self.rpc_calls_made}")
        print("=" * 50)

        # Log final summary
        self.logger.info(f"FINAL SUMMARY - Duration: {hours_running:.2f}h, Total Bets: {total_bets}, RPC Calls: {self.rpc_calls_made}")

    def show_summary(self):
        """Show periodic summary statistics"""
        current_time = time.time()
        session_duration = current_time - self.session_start_time
        hours_running = session_duration / 3600

        timestamp = datetime.now().strftime("%H:%M:%S")
        current_bet_count = len(self.current_mempool_bets)

        print(f"\n [{timestamp}] === TACOCLICKER MONITORING SUMMARY ===")
        print(f"     Session Duration: {hours_running:.1f} hours")
        print(f"    Current Next-Block Bets: {current_bet_count}")
        print(f"    Total Unique Bets Seen: {self.total_unique_bets_seen}")

        if hours_running > 0:
            rate_per_hour = self.total_unique_bets_seen / hours_running
            print(f"    Discovery Rate: {rate_per_hour:.1f} bets/hour")

        # Show hourly stats and reset
        if self.hourly_new_bets > 0 or self.hourly_removed_bets > 0:
            print(f"    Last Hour: +{self.hourly_new_bets} new, -{self.hourly_removed_bets} removed")

        # Show block analysis stats
        stats = self.mined_vs_expected_stats
        if stats['total_blocks_analyzed'] > 0:
            print(f"    Blocks Analyzed: {stats['total_blocks_analyzed']}")
            print(f"    Prediction Accuracy: {stats['accuracy_rate']:.1f}%")

        print(f"    RPC Calls Made: {self.rpc_calls_made}")
        print("=" * 60)
        print()

        # Reset hourly counters
        self.hourly_new_bets = 0
        self.hourly_removed_bets = 0

        # Log summary
        self.logger.info(f"SUMMARY - Duration: {hours_running:.1f}h, Current: {current_bet_count}, Total Seen: {self.total_unique_bets_seen}, Accuracy: {stats['accuracy_rate']:.1f}%")

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'session'):
            self.session.close()

    def check_specific_transaction(self, txid: str):
        """Check if a specific transaction is a TACOCLICKER bet"""
        print(f" Checking transaction: {txid}")

        try:
            # Get transaction details
            tx_details = self.get_transaction_details(txid)

            if not tx_details:
                print(f" Transaction {txid} not found or not accessible")
                return False

            print(f" Transaction found:")
            print(f"   Size: {tx_details.get('size', 0)} bytes")
            print(f"   VSize: {tx_details.get('vsize', 0)} bytes")
            print(f"   Confirmations: {tx_details.get('confirmations', 0)}")

            # Check if it's a TACOCLICKER bet
            is_bet = self.is_tacoclicker_bet(tx_details)

            if is_bet:
                print(f" TACOCLICKER BET DETECTED!")
                op_return_data = self.get_op_return_data(tx_details)
                sender_address = self.get_sender_address(txid)

                print(f"   Sender: {sender_address}")
                print(f"   OP_RETURN: {op_return_data}")

                # Log the bet
                source = "MEMPOOL" if tx_details.get('confirmations', 0) == 0 else "BLOCK"
                self.log_bet_transaction(txid, tx_details, source)

            else:
                print(f" Not a TACOCLICKER bet transaction")

                # Show OP_RETURN data for debugging with pattern analysis
                for i, output in enumerate(tx_details.get('vout', [])):
                    script_hex = output.get('scriptpubkey', '')
                    if script_hex.startswith('6a'):
                        runestone_data = self.decode_runestone_data(script_hex)
                        raw_data = runestone_data.get('raw_data', '')

                        print(f"   Output {i} OP_RETURN: {raw_data}")
                        print(f"   Length: {runestone_data.get('length', 0)} bytes")

                        # Detailed pattern analysis
                        analysis = self.analyze_tacoclicker_pattern(raw_data)
                        print(f"   Pattern Analysis:")
                        print(f"     Confidence Score: {analysis['confidence']}/100")
                        print(f"     Patterns Found: {', '.join(analysis['patterns_found'])}")
                        print(f"     Is Bet (≥70): {analysis['is_bet']}")

            return is_bet

        except Exception as e:
            print(f" Error checking transaction {txid}: {e}")
            return False

def main():
    """Main function"""
    import sys

    print("=== TACOCLICKER MEMPOOL MONITOR ===")

    args = sys.argv[1:]

    # Check if specific transaction ID provided
    if len(args) > 0 and len(args[0]) == 64:  # Bitcoin txid length
        txid = args[0]
        print(f"Checking specific transaction: {txid}")
        print()

        try:
            monitor = TacoClickerMempoolMonitor(enable_telegram=False)
            monitor.check_specific_transaction(txid)
        except Exception as e:
            print(f" Failed to check transaction: {e}")
            return 1
        finally:
            if 'monitor' in locals():
                monitor.cleanup()
        return 0

    # Flags
    enable_telegram = ('--telegram' in args)
    simulation_mode = ('--simulate' in args)
    if enable_telegram:
        args.remove('--telegram')
        print("Telegram notifications enabled")
    if simulation_mode:
        args.remove('--simulate')
        print("Simulation mode enabled (synthetic bet flows)")

    print("Real-time detection of TACOCLICKER bet transactions")
    print("Automatic summaries every minute")
    print("Press Ctrl+C to stop and see final summary")
    print()

    # Parse positional arguments
    mempool_interval = 2  # Default 2 seconds for mempool scans
    block_interval = 2     # Default 2 seconds for block checks

    if len(args) > 0:
        try:
            mempool_interval = int(args[0])
        except ValueError:
            print("Invalid mempool interval, using default 2 seconds")

    if len(args) > 1:
        try:
            block_interval = int(args[1])
        except ValueError:
            print("Invalid block interval, using default 2 seconds")

    try:
        monitor = TacoClickerMempoolMonitor(enable_telegram=enable_telegram, simulation_mode=simulation_mode)
        monitor.monitor_realtime(mempool_interval, block_interval)
    except Exception as e:
        print(f"Failed to start monitor: {e}")
        return 1
    finally:
        if 'monitor' in locals():
            monitor.cleanup()

    return 0

if __name__ == "__main__":
    exit(main())
