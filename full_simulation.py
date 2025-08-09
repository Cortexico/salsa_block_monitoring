#!/usr/bin/env python3
"""
Complete TACOCLICKER simulation that generates all notification scenarios:
1. Mempool bet flows (salsa context: 20-50 bets, normal: 0-2 bets)
2. Bet removals (falling behind due to fees)
3. Normal block mined notifications
4. Salsa block mined notifications with full analysis
5. Realistic timing and batching

This script simulates specific blocks (909072 salsa, 909073+ normal) and generates
all the Telegram messages you would see in production.

Usage: python full_simulation.py --telegram
"""

import time
import random
import secrets
from datetime import datetime
from typing import Set

from tacoclicker_mempool_monitor import TacoClickerMempoolMonitor
from salsa_block_analyzer import SalsaBlockAnalyzer


class FullTacoClickerSimulation:
    """Complete simulation of TACOCLICKER scenarios with realistic timing"""
    
    def __init__(self, enable_telegram: bool = True):
        self.enable_telegram = enable_telegram
        self.monitor = TacoClickerMempoolMonitor(
            enable_telegram=enable_telegram, 
            simulation_mode=True
        )
        self.analyzer = SalsaBlockAnalyzer()
        
        # Simulation state
        self.current_simulated_height = 909071  # Start just before salsa block 909072
        self.simulation_start_time = time.time()
        self.last_block_time = time.time()
        self.last_mempool_event = time.time()
        
        # Timing controls
        self.mempool_event_interval = 8.0  # seconds between mempool changes
        self.block_mining_interval = 45.0  # seconds between blocks (faster than real)
        
        print("🧪 Full TACOCLICKER Simulation Starting")
        print(f"📡 Telegram notifications: {'ENABLED' if enable_telegram else 'DISABLED'}")
        print(f"🎯 Starting at simulated block {self.current_simulated_height}")
        print(f"🌶️ Next salsa block: {self.current_simulated_height + 1}")
        print()

    def is_salsa_block(self, height: int) -> bool:
        """Check if block is a salsa block"""
        base_salsa = 908352
        return (height - base_salsa) % 144 == 0 and height >= base_salsa

    def get_next_salsa_block(self, current_height: int) -> int:
        """Get the next salsa block height"""
        base_salsa = 908352
        blocks_since_base = current_height - base_salsa
        blocks_to_next = 144 - (blocks_since_base % 144)
        if blocks_to_next == 144:
            blocks_to_next = 144  # Current block is a salsa block, next is 144 blocks later
        return current_height + blocks_to_next

    def simulate_mempool_activity(self):
        """Simulate realistic mempool bet flows"""
        current_time = time.time()
        
        if current_time - self.last_mempool_event < self.mempool_event_interval:
            return
            
        # Determine context (salsa vs normal)
        next_salsa = self.get_next_salsa_block(self.current_simulated_height)
        # Only show salsa context when the NEXT block will be salsa (exactly 1 block away)
        is_salsa_context = (next_salsa - self.current_simulated_height) == 1

        # Sometimes simulate removal-only scenarios to test decreases
        removal_only_chance = random.random() < 0.15  # 15% chance for removal-only

        if removal_only_chance and len(self.monitor.simulated_bets) >= 3:
            # Removal-only scenario: add 0, remove 1-3
            add_count = 0
            remove_count = random.randint(1, min(3, len(self.monitor.simulated_bets)))
            print(f"🔻 REMOVAL-ONLY scenario triggered")
        elif is_salsa_context:
            # Salsa context: 20-50 bets with some variation
            base_add = random.randint(15, 35)
            variation = random.randint(-5, 15)
            add_count = max(0, base_add + variation)
            remove_count = random.randint(0, max(1, add_count // 8))  # Small removals
        else:
            # Normal context: 0-2 bets
            add_count = random.randint(0, 2)
            remove_count = random.randint(0, max(1, len(self.monitor.simulated_bets) // 4))
        
        # Add new bets
        for _ in range(add_count):
            txid = secrets.token_hex(32)
            self.monitor.simulated_bets.add(txid)
        
        # Remove some bets (simulate falling behind)
        existing_bets = list(self.monitor.simulated_bets)
        for txid in existing_bets[:remove_count]:
            self.monitor.simulated_bets.discard(txid)
        
        self.last_mempool_event = current_time
        
        # Force a mempool scan to trigger notifications
        self.monitor.last_mempool_check = 0  # Force immediate scan
        
        print(f"🔄 Simulated mempool event: +{add_count} bets, -{remove_count} bets")
        print(f"   Context: {'🌶️ SALSA' if is_salsa_context else '⛏️ NORMAL'}")
        print(f"   Total simulated bets: {len(self.monitor.simulated_bets)}")

    def simulate_block_mining(self):
        """Simulate block mining with realistic scenarios"""
        current_time = time.time()
        
        if current_time - self.last_block_time < self.block_mining_interval:
            return
            
        # Mine the next block
        self.current_simulated_height += 1
        is_salsa = self.is_salsa_block(self.current_simulated_height)
        
        print(f"\n🎯 SIMULATING BLOCK {self.current_simulated_height} MINED")
        print(f"   Type: {'🌶️ SALSA BLOCK' if is_salsa else '⛏️ NORMAL BLOCK'}")
        
        if is_salsa:
            # Simulate salsa block mined
            print(f"🌶️ SALSA BLOCK {self.current_simulated_height} - Running full analysis...")

            # Send initial salsa block notification
            if self.monitor.telegram_bot:
                try:
                    self.monitor.telegram_bot.send_salsa_block_alert(self.current_simulated_height)
                    print(f"📱 Sent salsa block mined notification for block {self.current_simulated_height}")
                except Exception as e:
                    print(f"❌ Telegram salsa block notification failed: {e}")

            # Run comprehensive salsa block analysis
            try:
                salsa_results = self.monitor.run_salsa_block_analysis(self.current_simulated_height)
                if salsa_results and self.monitor.telegram_bot:
                    self.monitor.telegram_bot.send_salsa_block_results(salsa_results)
                    print(f"📱 Sent salsa block analysis results for block {self.current_simulated_height}")
                elif not salsa_results:
                    print(f"⚠️  No salsa analysis results generated for block {self.current_simulated_height}")
                elif not self.monitor.telegram_bot:
                    print(f"⚠️  No Telegram bot available for salsa results")
            except Exception as e:
                print(f"❌ Salsa block analysis failed: {e}")
        else:
            # Simulate normal block mined
            expected_count = len(self.monitor.expected_bets_for_next_block)
            actual_count = random.randint(max(0, expected_count - 1), expected_count + 1)
            accuracy = (min(expected_count, actual_count) / max(expected_count, 1)) * 100
            
            print(f"⛏️ Normal block analysis: Expected {expected_count}, Actual {actual_count}")
            
            # Send block analysis
            if self.monitor.telegram_bot and (expected_count > 0 or actual_count > 0):
                try:
                    self.monitor.telegram_bot.send_block_analysis(
                        self.current_simulated_height, expected_count, actual_count, accuracy
                    )
                    print("📤 Sent Telegram block analysis")
                except Exception as e:
                    print(f"❌ Telegram block analysis failed: {e}")
        
        # Reset mempool state after block (like real monitor)
        self.monitor.current_mempool_bets.clear()
        self.monitor.expected_bets_for_next_block.clear()
        self.monitor.last_bet_count = 0
        self.monitor._batch_accumulator['new_bets'].clear()
        self.monitor._batch_accumulator['removed_bets'].clear()
        self.monitor._batch_accumulator['last_reported_count'] = 0
        self.monitor.last_mempool_check = time.time()
        
        # Clear simulated bets for next cycle
        self.monitor.simulated_bets.clear()
        
        self.last_block_time = current_time
        print()

    def run_simulation(self, duration_minutes: int = 5):
        """Run the complete simulation for specified duration"""
        print(f"🚀 Starting {duration_minutes}-minute simulation...")
        print("📱 Watch your Telegram for all notification types!")
        print("🛑 Press Ctrl+C to stop early")
        print()
        
        # Set up Telegram bot
        if self.enable_telegram:
            next_salsa = self.get_next_salsa_block(self.current_simulated_height)
            self.monitor.setup_telegram_bot(next_salsa)
        
        end_time = time.time() + (duration_minutes * 60)
        
        try:
            while time.time() < end_time:
                current_time = time.time()
                
                # Simulate mempool activity
                self.simulate_mempool_activity()
                
                # Process mempool changes (triggers Telegram notifications)
                if current_time - self.monitor.last_mempool_check >= 5:  # Check every 5 seconds
                    bet_changes = self.monitor.scan_mempool_bet_changes()
                    
                    # Check batch window and send notifications
                    time_since_last_batch = current_time - self.monitor.last_mempool_check
                    if time_since_last_batch >= self.monitor.batch_window_seconds:
                        accumulated_new = len(self.monitor._batch_accumulator['new_bets'])
                        accumulated_removed = len(self.monitor._batch_accumulator['removed_bets'])
                        
                        if self.monitor.telegram_bot and (accumulated_new > 0 or accumulated_removed > 0):
                            try:
                                next_salsa = self.get_next_salsa_block(self.current_simulated_height)
                                batch_previous_count = bet_changes['current_count'] - accumulated_new + accumulated_removed
                                
                                self.monitor.telegram_bot.send_bet_count_update(
                                    bet_changes['current_count'], batch_previous_count,
                                    accumulated_new, accumulated_removed,
                                    next_salsa, self.current_simulated_height
                                )
                                print(f"📤 Sent Telegram batch: {bet_changes['current_count']} bets (+{accumulated_new}, -{accumulated_removed})")
                            except Exception as e:
                                print(f"❌ Telegram failed: {e}")
                        
                        # Reset batch
                        self.monitor._batch_accumulator['new_bets'].clear()
                        self.monitor._batch_accumulator['removed_bets'].clear()
                        self.monitor.last_mempool_check = current_time
                
                # Simulate block mining
                self.simulate_block_mining()
                
                # Sleep briefly
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n🛑 Simulation stopped by user")
        
        print(f"\n✅ Simulation completed!")
        print(f"📊 Final stats:")
        print(f"   Simulated blocks: {self.current_simulated_height - 909071}")
        print(f"   Duration: {(time.time() - self.simulation_start_time)/60:.1f} minutes")


def main():
    import sys
    
    enable_telegram = '--telegram' in sys.argv
    duration = 5  # Default 5 minutes
    
    if '--duration' in sys.argv:
        try:
            idx = sys.argv.index('--duration')
            duration = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Invalid duration, using 5 minutes")
    
    simulation = FullTacoClickerSimulation(enable_telegram=enable_telegram)
    simulation.run_simulation(duration_minutes=duration)


if __name__ == "__main__":
    main()
