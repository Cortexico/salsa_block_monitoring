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
        self.subscribers = []  # Changed to list to support topic objects
        self.load_subscribers()

    def load_subscribers(self):
        """Load subscribers from file"""
        try:
            with open("telegram_subscribers.json", "r") as f:
                data = json.load(f)
                raw_subscribers = data.get("subscribers", [])

                # Handle both old format (strings) and new format (objects)
                self.subscribers = []
                for sub in raw_subscribers:
                    if isinstance(sub, str):
                        # Old format: just chat_id
                        self.subscribers.append({"chat_id": sub})
                    elif isinstance(sub, dict):
                        # New format: object with chat_id and optional topic_id
                        self.subscribers.append(sub)

                print(f"Loaded {len(self.subscribers)} subscribers")
        except FileNotFoundError:
            self.subscribers = []
            print("No subscribers file found, starting fresh")
        except Exception as e:
            print(f"Error loading subscribers: {e}")
            self.subscribers = []

    def save_subscribers(self):
        """Save subscribers to file"""
        try:
            data = {"subscribers": self.subscribers}
            with open("telegram_subscribers.json", "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving subscribers: {e}")

    def add_subscriber(self, chat_id: str, topic_id: int = None):
        """Add a new subscriber with optional topic support"""
        subscriber = {"chat_id": str(chat_id)}
        if topic_id is not None:
            subscriber["topic_id"] = topic_id

        # Check if subscriber already exists
        for existing in self.subscribers:
            if (existing["chat_id"] == subscriber["chat_id"] and
                existing.get("topic_id") == subscriber.get("topic_id")):
                print(f"Subscriber already exists: {chat_id}" + (f" (topic {topic_id})" if topic_id else ""))
                return

        self.subscribers.append(subscriber)
        self.save_subscribers()
        topic_info = f" (topic {topic_id})" if topic_id else ""
        print(f"Added subscriber: {chat_id}{topic_info}")

    def send_message_sync(self, chat_id: str, message: str, topic_id: int = None) -> bool:
        """Send a message synchronously using requests with optional topic support"""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": message
            }

            # Add topic support for supergroups with topics
            if topic_id is not None:
                data["message_thread_id"] = topic_id

            topic_info = f" (topic {topic_id})" if topic_id else ""
            print(f"Sending message to {chat_id}{topic_info}...")

            response = requests.post(url, json=data, timeout=10)

            if response.status_code == 200:
                print(f"Message sent successfully to {chat_id}{topic_info}")
                return True
            else:
                print(f"Failed to send message to {chat_id}{topic_info}: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                return False

        except Exception as e:
            topic_info = f" (topic {topic_id})" if topic_id else ""
            print(f"Exception sending message to {chat_id}{topic_info}: {e}")
            return False

    def send_to_all_subscribers(self, message: str) -> int:
        """Send message to all subscribers"""
        if not self.subscribers:
            print("No subscribers to send message to")
            return 0

        sent_count = 0
        failed_subscribers = []

        for subscriber in self.subscribers.copy():
            chat_id = subscriber["chat_id"]
            topic_id = subscriber.get("topic_id")

            if self.send_message_sync(chat_id, message, topic_id):
                sent_count += 1
            else:
                failed_subscribers.append(subscriber)

        # Remove failed subscribers (blocked/deleted)
        for failed_sub in failed_subscribers:
            if failed_sub in self.subscribers:
                self.subscribers.remove(failed_sub)

        if failed_subscribers:
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
            # Calculate blocks left until next salsa block
            blocks_left = (next_salsa - current_block) if current_block is not None else None
            if blocks_left is not None:
                message_parts.append(f"🌶️ Next salsa block: {next_salsa} ({blocks_left} blocks left)\n{header_emoji} Taco Clicker Bets: {current_count}{header_suffix}")
            else:
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
        for subscriber in bot.subscribers:
            chat_id = subscriber["chat_id"]
            topic_id = subscriber.get("topic_id")
            topic_info = f" (topic {topic_id})" if topic_id else ""
            print(f"  - {chat_id}{topic_info}")

    print()
    print("For regular chats/channels: just enter the chat ID")
    print("For topic-specific messaging: enter chat_id:topic_id (e.g., -1001234567890:1)")
    print()

    user_input = input("Enter chat ID (or chat_id:topic_id for topics, or 'quit' to exit): ").strip()

    if user_input.lower() == 'quit':
        return

    if user_input:
        # Parse input for topic support
        if ':' in user_input:
            try:
                chat_id, topic_id = user_input.split(':', 1)
                topic_id = int(topic_id)
            except ValueError:
                print("Invalid format. Use chat_id:topic_id (e.g., -1001234567890:1)")
                return
        else:
            chat_id = user_input
            topic_id = None

        bot.add_subscriber(chat_id, topic_id)

        # Send test message
        topic_info = f" to topic {topic_id}" if topic_id else ""
        test_message = f"Welcome to TACOCLICKER notifications! You're now subscribed{topic_info}."
        if bot.send_message_sync(chat_id, test_message, topic_id):
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
