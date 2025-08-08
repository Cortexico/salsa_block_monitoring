#!/usr/bin/env python3
"""
Simple TACOCLICKER Telegram Bot - Notification Only
Sends notifications to subscribers without complex command handling
"""

import asyncio
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Set
from pathlib import Path

class SimpleTelegramBot:
    """Simple notification-only Telegram bot"""

    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self.subscribers = set()
        self.load_subscribers()

    def load_subscribers(self):
        """Load subscribers from file"""
        try:
            with open("telegram_subscribers.json", "r") as f:
                data = json.load(f)
                self.subscribers = set(data.get("subscribers", []))
                print(f"Loaded {len(self.subscribers)} subscribers")
        except FileNotFoundError:
            self.subscribers = set()
            print("No subscribers file found, starting fresh")
        except Exception as e:
            print(f"Error loading subscribers: {e}")
            self.subscribers = set()

    def save_subscribers(self):
        """Save subscribers to file"""
        try:
            data = {"subscribers": list(self.subscribers)}
            with open("telegram_subscribers.json", "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving subscribers: {e}")

    def add_subscriber(self, chat_id: str):
        """Add a new subscriber"""
        self.subscribers.add(str(chat_id))
        self.save_subscribers()
        print(f"Added subscriber: {chat_id}")

    def send_message_sync(self, chat_id: str, message: str) -> bool:
        """Send a message synchronously using requests"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": message
            }
            
            response = requests.post(url, json=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Failed to send message to {chat_id}: {e}")
            return False

    def send_to_all_subscribers(self, message: str) -> int:
        """Send message to all subscribers"""
        if not self.subscribers:
            print("No subscribers to send message to")
            return 0
        
        sent_count = 0
        failed_chats = []
        
        for chat_id in self.subscribers.copy():
            if self.send_message_sync(chat_id, message):
                sent_count += 1
            else:
                failed_chats.append(chat_id)
        
        # Remove failed chats (blocked/deleted)
        for chat_id in failed_chats:
            self.subscribers.discard(chat_id)
        
        if failed_chats:
            self.save_subscribers()
        
        return sent_count

    def send_bet_count_update(self, current_count: int, previous_count: int, new_bets: int, removed_bets: int, next_salsa: int, current_block: int = None) -> bool:
        """Send bet count variation notification in compact, readable format.
        Header uses 🌶️ when the next block is the salsa block, otherwise 🌮.
        """
        # Determine context (salsa vs normal)
        # Only show salsa context when the NEXT block will be salsa (exactly 1 block away)
        blocks_until_salsa = (next_salsa - current_block) if current_block is not None else None
        is_salsa_context = blocks_until_salsa is not None and blocks_until_salsa == 1
        header_emoji = "🌮"  # Always use taco emoji for bet count line

        # Build header with next-block number first
        if current_block is not None:
            next_block = current_block + 1
            prefix = f"⛏️ Next block: {next_block}" if is_salsa_context else f"⛏️ Next block: {next_block}"
        else:
            prefix = "⛏️ Next block: unknown"

        # Add salsa warning if next block is salsa
        if is_salsa_context:
            message_parts = ["🌶️SALSA BLOCK INCOMING🌶️", prefix]
        else:
            message_parts = [prefix]

        # Change indicator
        net_change = current_count - previous_count
        if net_change > 0:
            header_suffix = f" ({'📈'} +{net_change})"
        elif net_change < 0:
            header_suffix = f" ({'📉'} {net_change})"
        else:
            header_suffix = ""

        # Message body
        if not is_salsa_context:
            message_parts.append(f"🌶️ Next salsa block: {next_salsa}\n{header_emoji} Taco Clicker Bets: {current_count}{header_suffix}")
        else:
            message_parts.append(f"{header_emoji} Taco Clicker Bets: {current_count}{header_suffix}")
        if new_bets > 0:
            message_parts.append(f"➕ New: {new_bets}")
        if removed_bets > 0:
            message_parts.append(f"➖ Removed: {removed_bets}")

        message = "\n".join(message_parts)

        sent_count = self.send_to_all_subscribers(message)
        return sent_count > 0

    def send_block_analysis(self, block_height: int, expected: int, actual: int, accuracy: float) -> bool:
        """Send block analysis notification (no timestamp, 🌮 header, grammar-aware)."""
        # Grammar-aware labels
        bet_word_expected = "bet" if expected == 1 else "bets"
        bet_word_actual = "bet" if actual == 1 else "bets"

        message = f"⛏️ Block {block_height} Mined\n"
        message += f"   Expected: {expected} {bet_word_expected}\n"
        message += f"   Actual: {actual} {bet_word_actual}\n"
        message += f"   Accuracy: {accuracy:.1f}%"

        sent_count = self.send_to_all_subscribers(message)
        return sent_count > 0

    def send_salsa_block_notification(self, block_height: int) -> bool:
        """Send salsa block notification"""
        message = f"🌶️ SALSA BLOCK {block_height} MINED! ⛏️"
        sent_count = self.send_to_all_subscribers(message)
        return sent_count > 0

    def send_salsa_block_results(self, salsa_summary: str) -> bool:
        """Send detailed salsa block results"""
        sent_count = self.send_to_all_subscribers(salsa_summary)
        return sent_count > 0

    def send_startup_message(self, next_salsa: int) -> bool:
        """Send startup notification"""
        message = f"🌶️ Salsabot Started → Next Salsa Block: {next_salsa}"

        sent_count = self.send_to_all_subscribers(message)
        return sent_count > 0

class SimpleTelegramIntegration:
    """Simple integration for mempool monitor"""
    
    def __init__(self, bot_token: str):
        self.bot = SimpleTelegramBot(bot_token)
    
    def start_bot(self, next_salsa: int) -> bool:
        """Start the bot (just send startup message)"""
        try:
            if self.bot.subscribers:
                self.bot.send_startup_message(next_salsa)
            
            print(f"Simple Telegram bot ready with {len(self.bot.subscribers)} subscribers")
            print("To add subscribers, they need to send their chat ID to you manually")
            print("or use the full bot with command handlers")
            return True
        except Exception as e:
            print(f"Failed to start simple Telegram bot: {e}")
            return False
    
    def send_bet_count_update(self, current_count: int, previous_count: int, new_bets: int, removed_bets: int, next_salsa: int, current_block: int = None) -> bool:
        """Send bet count update"""
        return self.bot.send_bet_count_update(current_count, previous_count, new_bets, removed_bets, next_salsa, current_block)

    def send_block_analysis(self, block_height: int, expected: int, actual: int, accuracy: float) -> bool:
        """Send block analysis"""
        return self.bot.send_block_analysis(block_height, expected, actual, accuracy)
    
    def send_salsa_block_alert(self, block_height: int) -> bool:
        """Send salsa block alert"""
        return self.bot.send_salsa_block_notification(block_height)

    def send_salsa_block_results(self, salsa_summary: str) -> bool:
        """Send salsa block results"""
        return self.bot.send_salsa_block_results(salsa_summary)
    
    def cleanup(self):
        """Cleanup (nothing needed for simple bot)"""
        pass

def load_telegram_config(config_file: str = "telegram_config.json") -> Dict:
    """Load Telegram bot configuration"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            if 'bot_token' not in config:
                raise ValueError("bot_token not found in config file")
            return config
    except FileNotFoundError:
        print(f"Config file {config_file} not found!")
        print("Please create the config file with your bot token:")
        print(f'{{ "bot_token": "YOUR_BOT_TOKEN_HERE" }}')
        raise
    except Exception as e:
        print(f"Error loading config: {e}")
        raise

def add_subscriber_manually():
    """Helper function to manually add a subscriber"""
    print("=== ADD TELEGRAM SUBSCRIBER ===")
    print()

    # Load config to get bot token
    try:
        config = load_telegram_config()
        bot = SimpleTelegramBot(config['bot_token'])
    except Exception as e:
        print(f"Failed to load config: {e}")
        return
    
    print(f"Current subscribers: {len(bot.subscribers)}")
    if bot.subscribers:
        for chat_id in bot.subscribers:
            print(f"  - {chat_id}")
    
    print()
    chat_id = input("Enter chat ID to add (or 'quit' to exit): ").strip()
    
    if chat_id.lower() == 'quit':
        return
    
    if chat_id:
        bot.add_subscriber(chat_id)
        
        # Send test message
        test_message = "Welcome to TACOCLICKER notifications! You're now subscribed."
        if bot.send_message_sync(chat_id, test_message):
            print("Test message sent successfully!")
        else:
            print("Failed to send test message")

def main():
    """Test the simple bot"""
    print("=== SIMPLE TELEGRAM BOT TEST ===")
    print()
    
    # Load config
    config = load_telegram_config()
    
    # Create bot
    bot = SimpleTelegramIntegration(config['bot_token'])
    
    # Test startup
    if bot.start_bot(908496):
        print("Bot started successfully!")
        
        # Send test notifications if there are subscribers
        if bot.bot.subscribers:
            print("Sending test notifications...")
            
            bot.send_bet_alert(42, 908496)
            time.sleep(1)
            bot.send_batch_alert(5, 47, 908496)
            time.sleep(1)
            bot.send_salsa_block_alert(908352)
            
            print("Test notifications sent!")
        else:
            print("No subscribers yet. Use add_subscriber_manually() to add some.")
    else:
        print("Failed to start bot")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "add":
        add_subscriber_manually()
    else:
        main()
