import discord
from discord.ext import commands, tasks
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import json
import os
from collections import defaultdict, Counter
from statistics import mean, median
from nrclex import NRCLex
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import emoji
import re

intents = discord.Intents.default()

bot = discord.Bot()

# Create data directory structure
DATA_DIR = "guild_data"

# Add this import at the top with your other imports
import numpy as np


# Add this function to load the sentiment model (put it near your other helper functions)
def load_sentiment_model():
    """Load RoBERTa model for sentiment analysis"""
    try:
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        print(f"Error loading sentiment model: {e}")
        return None, None


def analyze_sentiment_roberta(text, tokenizer, model):
    """Analyze sentiment using RoBERTa model"""
    if not tokenizer or not model:
        return None

    try:
        # Clean text - remove mentions, links, etc.
        cleaned_text = re.sub(r'<@[!&]?\d+>', '', text)  # Remove mentions
        cleaned_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                              cleaned_text)  # Remove URLs
        cleaned_text = emoji.demojize(cleaned_text)  # Convert emojis to text

        if len(cleaned_text.strip()) == 0:
            return None

        # Tokenize and get prediction
        encoded_text = tokenizer(cleaned_text, return_tensors='pt', truncation=True, max_length=512)

        with torch.no_grad():
            output = model(**encoded_text)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)

        # Labels: 0=negative, 1=neutral, 2=positive
        sentiment_score = scores[2] - scores[0]  # positive - negative
        return sentiment_score

    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return None

@bot.slash_command(name="comparesentiment", description="Analyze sentiment of users over a specified timeframe")
@discord.option("days", description="Number of days to analyze", default=1, min_value=1, max_value=30)
@discord.option("min_messages", description="Minimum messages needed to be included", default=5, min_value=1, max_value=50)
async def comparesentiment(ctx: discord.ApplicationContext, days: int, min_messages: int):
    await ctx.defer()

    # Load sentiment model
    tokenizer, model = load_sentiment_model()
    if not tokenizer or not model:
        await ctx.respond("Error loading sentiment analysis model. Please try again later.")
        return

    # Get the specified date range
    now = datetime.now(timezone.utc)

    # For 1 day: get yesterday's full day
    # For multiple days: get the last N complete days
    if days == 1:
        # Just yesterday
        yesterday = now - timedelta(days=1)
        start_of_period = datetime(yesterday.year, yesterday.month, yesterday.day, tzinfo=timezone.utc)
        end_of_period = now
        date_range_text = yesterday.strftime('%Y-%m-%d')
    else:
        # Last N days (excluding today to avoid partial day data)
        end_date = now  # Start of today
        start_date = end_date - timedelta(days=days)
        start_of_period = start_date
        end_of_period = end_date
        date_range_text = f"{start_date.strftime('%Y-%m-%d')} to {(end_date).strftime('%Y-%m-%d')}"

    # Collect messages from the specified time period
    user_messages = defaultdict(list)
    total_messages = 0

    timeframe_text = f"last {days} day{'s' if days > 1 else ''}"
    await ctx.followup.send(f"Collecting messages from {date_range_text}...")

    print(f"DEBUG: Searching from {start_of_period} to {end_of_period}")  # Debug line

    for channel in ctx.guild.text_channels:
        if not channel.permissions_for(ctx.guild.me).read_message_history:
            continue

        try:
            message_count = 0
            async for message in channel.history(limit=None, after=start_of_period):
                if message.author.bot:
                    continue

                if len(message.content.strip()) > 0 :  # Only non-empty messages
                    user_messages[message.author.id].append(message.content)
                    total_messages += 1
                    message_count += 1

            if message_count > 0:
                print(f"DEBUG: Found {message_count} messages in #{channel.name}")  # Debug line

        except (discord.Forbidden, discord.HTTPException) as e:
            print(f"DEBUG: Error accessing #{channel.name}: {e}")  # Debug line
            continue

    print(f"DEBUG: Total messages found: {total_messages} from {len(user_messages)} users")  # Debug line

    if total_messages == 0:
        await ctx.followup.send(
            f"No messages found from {date_range_text}. Check if the bot has permission to read message history in channels.")
        return

    # Filter users with minimum message count
    filtered_users = {user_id: messages for user_id, messages in user_messages.items()
                      if len(messages) >= min_messages}

    if not filtered_users:
        await ctx.followup.send(f"No users found with at least {min_messages} messages from the {timeframe_text}.")
        return

    await ctx.followup.send(f"Analyzing sentiment for {len(filtered_users)} users...")

    # Analyze sentiment for each user
    user_sentiments = {}

    for user_id, messages in filtered_users.items():
        try:
            user = await bot.fetch_user(user_id)
            sentiments = []

            for message in messages:
                sentiment = analyze_sentiment_roberta(message, tokenizer, model)
                if sentiment is not None:
                    sentiments.append(sentiment)

            if sentiments:
                avg_sentiment = np.mean(sentiments)
                user_sentiments[user.display_name] = {
                    'sentiment': avg_sentiment,
                    'message_count': len(messages),
                    'analyzed_count': len(sentiments)
                }

        except discord.NotFound:
            continue
        except Exception as e:
            print(f"Error processing user {user_id}: {e}")
            continue

    if not user_sentiments:
        await ctx.followup.send("Could not analyze sentiment for any users.")
        return

    # Sort users by sentiment (most positive first)
    sorted_users = sorted(user_sentiments.items(), key=lambda x: x[1]['sentiment'], reverse=True)

    # Create bar chart
    plt.figure(figsize=(12, 8))
    names = [item[0] for item in sorted_users]
    sentiments = [item[1]['sentiment'] for item in sorted_users]

    # Color bars based on sentiment
    colors = ['green' if s > 0.1 else 'red' if s < -0.1 else 'gray' for s in sentiments]

    bars = plt.bar(names, sentiments, color=colors, alpha=0.7)

    # Create title based on timeframe
    plt.title(f"User Sentiment Analysis - {date_range_text}")
    plt.xlabel("Users")
    plt.ylabel("Average Sentiment Score")
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, sentiment in zip(bars, sentiments):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{sentiment:.3f}',
                 ha='center', va='bottom' if height > 0 else 'top')

    plt.tight_layout()

    image_path = f"{ctx.guild.id}_sentiment_{days}days.png"
    plt.savefig(image_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Create summary text
    most_positive = sorted_users[0]
    most_negative = sorted_users[-1]

    summary = f"**Sentiment Analysis - {date_range_text}**\n\n"
    summary += f"*Users analyzed:** {len(user_sentiments)} (with ≥{min_messages} messages)\n"
    summary += f"**Total messages:** {sum(data['message_count'] for data in user_sentiments.values())}\n"
    summary += f"**Timeframe:** {timeframe_text}\n\n"
    summary += f"**Most positive:** {most_positive[0]} ({most_positive[1]['sentiment']:.3f})\n"
    summary += f"**Most negative:** {most_negative[0]} ({most_negative[1]['sentiment']:.3f})\n\n"
    summary += f"*Sentiment scale: -1 (very negative) to +1 (very positive)*"

    await ctx.followup.send(content=summary, file=discord.File(image_path))
    os.remove(image_path)

def ensure_data_directory(guild_id, channel_id=None):
    """Ensure the data directory structure exists"""
    guild_dir = os.path.join(DATA_DIR, str(guild_id))
    os.makedirs(guild_dir, exist_ok=True)

    if channel_id:
        channel_dir = os.path.join(guild_dir, "channels")
        os.makedirs(channel_dir, exist_ok=True)

    return guild_dir


def get_channel_file_path(guild_id, channel_id):
    """Get the file path for a specific channel's data"""
    guild_dir = ensure_data_directory(guild_id, channel_id)
    return os.path.join(guild_dir, "channels", f"{channel_id}.json")


def get_guild_summary_path(guild_id):
    """Get the file path for guild summary data"""
    guild_dir = ensure_data_directory(guild_id)
    return os.path.join(guild_dir, "summary.json")


def load_channel_data(guild_id, channel_id):
    """Load data for a specific channel"""
    file_path = get_channel_file_path(guild_id, channel_id)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {
        "message_counts": {},
        "user_message_counts": {},
        "last_timestamp": None
    }


def save_channel_data(guild_id, channel_id, data):
    """Save data for a specific channel"""
    file_path = get_channel_file_path(guild_id, channel_id)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def load_guild_summary(guild_id):
    """Load guild summary data"""
    file_path = get_guild_summary_path(guild_id)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {
        "message_counts": {},
        "user_message_counts": {},
        "channels": {},
        "last_updated": None
    }


def save_guild_summary(guild_id, data):
    """Save guild summary data"""
    file_path = get_guild_summary_path(guild_id)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def get_all_guild_channels(guild_id):
    """Get list of all channels that have data files"""
    guild_dir = ensure_data_directory(guild_id)
    channels_dir = os.path.join(guild_dir, "channels")
    if not os.path.exists(channels_dir):
        return []

    channel_files = [f for f in os.listdir(channels_dir) if f.endswith('.json')]
    return [int(f[:-5]) for f in channel_files]  # Remove .json extension


def aggregate_guild_data(guild_id):
    """Aggregate data from all channels in a guild"""
    channel_ids = get_all_guild_channels(guild_id)

    guild_message_counts = defaultdict(int)
    guild_user_counts = defaultdict(lambda: defaultdict(int))

    for channel_id in channel_ids:
        channel_data = load_channel_data(guild_id, channel_id)

        # Aggregate message counts
        for date, count in channel_data["message_counts"].items():
            guild_message_counts[date] += count

        # Aggregate user message counts
        for user_id, user_dates in channel_data["user_message_counts"].items():
            for date, count in user_dates.items():
                guild_user_counts[user_id][date] += count

    return dict(guild_message_counts), dict(guild_user_counts)


async def build_or_update_history_data(guild: discord.Guild):
    print(f"[INIT] Updating historical data for {guild.name}")

    for channel in guild.text_channels:
        if not channel.permissions_for(guild.me).read_message_history:
            print(f"[SKIP] No permission to read {channel.name}")
            continue

        channel_data = load_channel_data(guild.id, channel.id)

        last_timestamp = None
        if channel_data["last_timestamp"]:
            last_timestamp = datetime.fromisoformat(channel_data["last_timestamp"])
            print(f"[INIT] Resuming {channel.name} from {last_timestamp}")
        else:
            print(f"[INIT] No existing data for {channel.name}, scanning full history")

        message_counts = defaultdict(int)
        user_message_counts = defaultdict(lambda: defaultdict(int))
        newest_timestamp = last_timestamp

        print(f"[SCAN] Starting {channel.name}...")
        count = 0
        try:
            async for msg in channel.history(limit=None, after=last_timestamp):
                if msg.created_at >= datetime.now(timezone.utc):
                    continue  # Skip future/today's messages

                date_key = msg.created_at.date().isoformat()
                message_counts[date_key] += 1

                user_id = str(msg.author.id)
                user_message_counts[user_id][date_key] += 1

                count += 1

                if (not newest_timestamp) or (msg.created_at > newest_timestamp):
                    newest_timestamp = msg.created_at

                if count % 1000 == 0:
                    print(f"[SCAN] {channel.name}: scanned {count} messages so far...")

            print(f"[SCAN] Finished {channel.name}: total {count} messages")

        except discord.Forbidden:
            print(f"[SKIP] Forbidden reading {channel.name}")
            continue
        except discord.HTTPException as e:
            print(f"[ERROR] HTTP exception in {channel.name}: {e}")
            continue

        # Merge new data with existing channel data
        for date, c in message_counts.items():
            existing = channel_data["message_counts"].get(date, 0)
            channel_data["message_counts"][date] = existing + c

        for user_id, user_dates in user_message_counts.items():
            if user_id not in channel_data["user_message_counts"]:
                channel_data["user_message_counts"][user_id] = {}
            for date, c in user_dates.items():
                existing = channel_data["user_message_counts"][user_id].get(date, 0)
                channel_data["user_message_counts"][user_id][date] = existing + c

        if newest_timestamp:
            channel_data["last_timestamp"] = (newest_timestamp + timedelta(seconds=1)).isoformat()

        save_channel_data(guild.id, channel.id, channel_data)
        print(f"[SAVE] Saved data for channel {channel.name}")

    # Update guild summary
    guild_msg_counts, guild_user_counts = aggregate_guild_data(guild.id)
    summary_data = {
        "message_counts": guild_msg_counts,
        "user_message_counts": guild_user_counts,
        "channels": {str(ch.id): ch.name for ch in guild.text_channels},
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
    save_guild_summary(guild.id, summary_data)

    print(f"[INIT] Finished history update for {guild.name}")


async def on_message(message: discord.Message):
    if message.author.bot:
        return

    now = datetime.now(timezone.utc)
    start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    date_str = start.date().isoformat()

    guild_id = message.guild.id
    channel_id = message.channel.id
    user_id = str(message.author.id)

    # Update channel data
    channel_data = load_channel_data(guild_id, channel_id)
    channel_data["message_counts"][date_str] = channel_data["message_counts"].get(date_str, 0) + 1

    if user_id not in channel_data["user_message_counts"]:
        channel_data["user_message_counts"][user_id] = {}

    channel_data["user_message_counts"][user_id][date_str] = (
            channel_data["user_message_counts"][user_id].get(date_str, 0) + 1
    )

    save_channel_data(guild_id, channel_id, channel_data)

    # Update guild summary (this could be optimized to batch updates)
    guild_msg_counts, guild_user_counts = aggregate_guild_data(guild_id)
    summary_data = load_guild_summary(guild_id)
    summary_data["message_counts"] = guild_msg_counts
    summary_data["user_message_counts"] = guild_user_counts
    summary_data["last_updated"] = now.isoformat()
    save_guild_summary(guild_id, summary_data)

    await bot.process_commands(message)


@tasks.loop(hours=24)
async def daily_message_counter():
    now = datetime.now(timezone.utc)
    start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)

    for guild in bot.guilds:
        total_messages = 0
        user_message_counts = defaultdict(int)

        for channel in guild.text_channels:
            if not channel.permissions_for(guild.me).read_message_history:
                continue

            channel_messages = 0
            channel_user_counts = defaultdict(int)

            try:
                async for msg in channel.history(limit=None, after=start):
                    channel_messages += 1
                    total_messages += 1
                    user_id = str(msg.author.id)
                    channel_user_counts[user_id] += 1
                    user_message_counts[user_id] += 1
            except discord.Forbidden:
                continue
            except discord.HTTPException:
                continue

            # Update channel data
            if channel_messages > 0:
                channel_data = load_channel_data(guild.id, channel.id)
                date_str = start.date().isoformat()

                channel_data["message_counts"][date_str] = (
                        channel_data["message_counts"].get(date_str, 0) + channel_messages
                )

                for user_id, count in channel_user_counts.items():
                    if user_id not in channel_data["user_message_counts"]:
                        channel_data["user_message_counts"][user_id] = {}
                    channel_data["user_message_counts"][user_id][date_str] = (
                            channel_data["user_message_counts"][user_id].get(date_str, 0) + count
                    )

                save_channel_data(guild.id, channel.id, channel_data)

        # Update guild summary
        if total_messages > 0:
            guild_msg_counts, guild_user_counts = aggregate_guild_data(guild.id)
            summary_data = {
                "message_counts": guild_msg_counts,
                "user_message_counts": guild_user_counts,
                "channels": {str(ch.id): ch.name for ch in guild.text_channels},
                "last_updated": now.isoformat()
            }
            save_guild_summary(guild.id, summary_data)

        print(f"[{start.date().isoformat()}] Logged {total_messages} messages for {guild.name}")


@bot.event
async def on_ready():
    print(f'Bot is online as {bot.user}!')
    for guild in bot.guilds:
        guild_summary = load_guild_summary(guild.id)
        if not guild_summary["message_counts"]:  # No existing data
            await build_or_update_history_data(guild)
    daily_message_counter.start()


@bot.slash_command(name="graph", description="Show message graph over time")
@discord.option("mode", description="Group data by time scale", choices=["day", "month", "year"], default="day")
@discord.option("cumulative", description="Show cumulative message total", default=False)
async def graph(ctx: discord.ApplicationContext, mode: str, cumulative: bool):
    await ctx.defer()

    guild_summary = load_guild_summary(ctx.guild.id)
    if not guild_summary["message_counts"]:
        await ctx.respond("No data available.")
        return

    x_labels, y_counts = prep_graph(guild_summary["message_counts"], mode, cumulative)

    plt.figure(figsize=(10, 5))
    plt.plot(x_labels, y_counts, marker="o")
    plt.title(f"{'Cumulative ' if cumulative else ''}Message Count Over Time ({mode.title()}s)")
    plt.xlabel(f"{mode.title()}s since first record")
    plt.ylabel("Total Messages" if cumulative else "Messages per Period")
    plt.xticks(rotation=45)
    plt.tight_layout()

    image_path = f"{ctx.guild.id}_message_graph.png"
    plt.savefig(image_path)
    plt.close()

    await ctx.respond(file=discord.File(image_path))
    os.remove(image_path)


@bot.slash_command(name="usergraph", description="Show message graph over time for different users")
@discord.option("user1", description="First user", type=discord.User, required=True)
@discord.option("user2", description="Second user", type=discord.User, required=False)
@discord.option("user3", description="Third user", type=discord.User, required=False)
@discord.option("user4", description="Fourth user", type=discord.User, required=False)
@discord.option("user5", description="Fifth user", type=discord.User, required=False)
@discord.option("mode", description="Group data by time scale", choices=["day", "month", "year"], default="day")
@discord.option("cumulative", description="Show cumulative message total", default=False)
async def usergraph(ctx: discord.ApplicationContext, user1: discord.User, user2: discord.User, user3: discord.User,
                    user4: discord.User, user5: discord.User, mode: str, cumulative: bool):
    await ctx.defer()

    guild_summary = load_guild_summary(ctx.guild.id)
    if not guild_summary["user_message_counts"]:
        await ctx.respond("No user message data available.")
        return

    users = [user1, user2, user3, user4, user5]
    all_series = {}

    for user in users:
        if user is None:
            continue
        user_counts = guild_summary["user_message_counts"].get(str(user.id))
        if not user_counts:
            await ctx.respond(f"No data available for user {user.name}.")
            return

        x_labels, y_counts = prep_graph(user_counts, mode, cumulative)
        all_series[user.name] = (x_labels, y_counts)

    if not all_series:
        await ctx.respond("No data available for any users.")
        return

    shared_x = sorted(set(x for x_labels, _ in all_series.values() for x in x_labels))
    user_lines = {}

    for name, (x_labels, y_counts) in all_series.items():
        data_map = dict(zip(x_labels, y_counts))
        aligned_counts = []
        last_value = 0

        for x in shared_x:
            if x in data_map:
                if cumulative:
                    last_value = data_map[x]
                else:
                    aligned_counts.append(data_map[x])
                    continue
            aligned_counts.append(last_value if cumulative else 0)

        user_lines[name] = aligned_counts

    plt.figure(figsize=(12, 6))
    for name, y_values in user_lines.items():
        plt.plot(shared_x, y_values, marker="o", label=name)

    plt.title(f"{'Cumulative ' if cumulative else ''}User Message Comparison ({mode.title()}s)")
    plt.xlabel(f"{mode.title()}s since first record")
    plt.ylabel("Messages")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    image_path = f"{ctx.guild.id}_user_message_graph.png"
    plt.savefig(image_path)
    plt.close()

    await ctx.respond(file=discord.File(image_path))
    os.remove(image_path)


def prep_graph(msg_counts: dict, mode: str, cumulative: bool):
    parsed = [(datetime.fromisoformat(date), count) for date, count in msg_counts.items()]
    parsed.sort()

    if mode == "day":
        grouped = parsed
        label_vals = [(dt - parsed[0][0]).days + 1 for dt, _ in grouped]  # start at 1
    else:
        counts_by_group = defaultdict(int)
        for dt, count in parsed:
            if mode == "month":
                key = dt.strftime("%Y-%m")
            elif mode == "year":
                key = dt.strftime("%Y")
            counts_by_group[key] += count

        grouped = [(datetime.strptime(k, "%Y-%m") if mode == "month" else datetime.strptime(k, "%Y"), v)
                   for k, v in counts_by_group.items()]
        grouped.sort()

        start = grouped[0][0]
        if mode == "month":
            label_vals = [(dt.year - start.year) * 12 + (dt.month - start.month) + 1 for dt, _ in grouped]
        else:
            label_vals = [(dt.year - start.year) + 1 for dt, _ in grouped]

    y_counts = [count for _, count in grouped]

    if cumulative:
        for i in range(1, len(y_counts)):
            y_counts[i] += y_counts[i - 1]

    return label_vals, y_counts


@bot.slash_command(name="maxday", description="Find the day with the most messages (all users)")
@discord.option("mode", description="Group by time", choices=["day", "month", "year"], default="day")
async def maxday(ctx: discord.ApplicationContext, mode: str):
    await ctx.defer()

    guild_summary = load_guild_summary(ctx.guild.id)
    if not guild_summary["message_counts"]:
        await ctx.respond("No data available")
        return

    label, count = get_max_message_day(guild_summary["message_counts"], mode)
    if not label:
        await ctx.respond("No messages found")
    else:
        await ctx.respond(f"Most messages in a {mode}: **{count}** on **{label}**.")


@bot.slash_command(name="usermax", description="Find the user's most active day")
@discord.option("user", description="The user to analyze", required=True)
@discord.option("mode", description="Group by time", choices=["day", "month", "year"], default="day")
async def usermax(ctx: discord.ApplicationContext, user: discord.User, mode: str):
    await ctx.defer()

    guild_summary = load_guild_summary(ctx.guild.id)
    if not guild_summary["user_message_counts"]:
        await ctx.respond("No per-user data available.")
        return

    user_data = guild_summary["user_message_counts"].get(str(user.id))
    if not user_data:
        await ctx.respond(f"No message data available for {user.name}.")
        return

    label, count = get_max_message_day(user_data, mode)
    if not label:
        await ctx.respond(f"No message data found for {user.name}.")
    else:
        await ctx.respond(f"**{user.name}**'s most active {mode} was **{label}** with **{count}** messages")


def get_max_message_day(counts_dict, mode="day"):
    grouped = defaultdict(int)
    for date_str, count in counts_dict.items():
        dt = datetime.fromisoformat(date_str)
        if mode == "day":
            key = dt.strftime("%Y-%m-%d")
        elif mode == "month":
            key = dt.strftime("%Y-%m")
        elif mode == "year":
            key = dt.strftime("%Y")
        grouped[key] += count

    if not grouped:
        return None, 0
    return max(grouped.items(), key=lambda item: item[1])


@bot.slash_command(name="stat", description="Get mean or median messages per day/month/year for the server")
@discord.option("type", choices=["mean", "median"])
@discord.option("mode", choices=["day", "month", "year"])
async def stat(ctx: discord.ApplicationContext, type: str, mode: str):
    await ctx.defer()
    guild_summary = load_guild_summary(ctx.guild.id)
    if not guild_summary["message_counts"]:
        await ctx.respond("No message data found.")
        return

    result = compute_stat_from_counts(guild_summary["message_counts"], mode, type)
    if result is None:
        await ctx.respond("Not enough data to compute statistic.")
    else:
        await ctx.respond(f"**{type.title()} messages per {mode}:** {result}")


@bot.slash_command(name="userstat", description="Get mean or median messages per time unit for a user")
@discord.option("user", description="User to analyze")
@discord.option("type", choices=["mean", "median"])
@discord.option("mode", choices=["day", "month", "year"])
async def userstat(ctx: discord.ApplicationContext, user: discord.Member, type: str, mode: str):
    await ctx.defer()
    guild_summary = load_guild_summary(ctx.guild.id)
    if not guild_summary["user_message_counts"]:
        await ctx.respond("No user message data found")
        return

    user_data = guild_summary["user_message_counts"].get(str(user.id))
    if not user_data:
        await ctx.respond(f"No message data found for {user.name}")
        return

    result = compute_stat_from_counts(user_data, mode, type)
    if result is None:
        await ctx.respond("Not enough data to compute")
    else:
        await ctx.respond(f"**{type.title()} messages per {mode} for {user.name}:** {result}")


def compute_stat_from_counts(counts_dict, mode, stat_type):
    parsed = [(datetime.fromisoformat(date), count) for date, count in counts_dict.items()]
    if not parsed:
        return None

    parsed.sort()
    start = parsed[0][0]
    end = parsed[-1][0]

    # Fill in missing periods
    filled_counts = defaultdict(int)

    if mode == "day":
        delta = timedelta(days=1)
        current = start
        while current <= end:
            key = current.date().isoformat()
            filled_counts[key] = 0
            current += delta

        for dt, count in parsed:
            filled_counts[dt.date().isoformat()] += count

    elif mode == "month":
        current = start.replace(day=1)
        while current <= end:
            key = current.strftime("%Y-%m")
            filled_counts[key] = 0
            # Advance by one month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)

        for dt, count in parsed:
            filled_counts[dt.strftime("%Y-%m")] += count

    elif mode == "year":
        current = datetime(start.year, 1, 1)
        while current <= end:
            key = str(current.year)
            filled_counts[key] = 0
            current = current.replace(year=current.year + 1)

        for dt, count in parsed:
            filled_counts[str(dt.year)] += count

    totals = list(filled_counts.values())
    if not totals:
        return None

    if stat_type == "mean":
        return round(mean(totals), 2)
    elif stat_type == "median":
        return round(median(totals), 2)
    return None


@bot.slash_command(name="emotions", description="sentiment analysis from user")
@discord.option("user", description="User to analyze", type=discord.User)
@discord.option("limit", description="Max messages to analyze", default=100, min_value=1, max_value=1000)
@discord.option("channel", description="Specific channel to search in (optional)", required=False)
async def emotions(ctx: discord.ApplicationContext, user: discord.User, limit: int,
                   channel: discord.TextChannel = None):
    await ctx.defer()
    messages = []

    channels_to_search = [channel] if channel else ctx.guild.text_channels

    for ch in channels_to_search:
        if not ch.permissions_for(ctx.guild.me).read_message_history:
            continue
        try:
            async for msg in ch.history(limit=None, oldest_first=False):
                if msg.author.id == user.id:
                    messages.append(msg.content)
                    if len(messages) >= limit:
                        break
        except discord.Forbidden:
            continue
        except discord.HTTPException:
            continue
        if len(messages) >= limit:
            break

    image_path, result = analyze_emotions(messages, limit_label=f"Last {len(messages)} messages", user=user)
    if image_path:
        await ctx.respond(content=result, file=discord.File(image_path))
        os.remove(image_path)
    else:
        await ctx.respond(result)


def analyze_emotions(messages, limit_label=None, user=None):
    emotion_counter = Counter()

    for msg in messages:
        text_obj = NRCLex(msg)
        for emotion in text_obj.raw_emotion_scores:
            emotion_counter[emotion] += text_obj.raw_emotion_scores[emotion]

    total = sum(emotion_counter.values())
    if total == 0:
        return None, "No detectable emotions found in the messages"

    # Pie chart
    labels = list(emotion_counter.keys())
    sizes = [emotion_counter[label] for label in labels]

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis("equal")
    plt.title(f"Emotion Distribution{f' ({limit_label})' if limit_label else ''} for {user.name}")
    output_file = "emotion_pie.png"
    plt.savefig(output_file)
    plt.close()

    return output_file, f"Total emotions: **{total}**"


# Add new command to show channel-specific stats
@bot.slash_command(name="channelstats", description="Show statistics for a specific channel")
@discord.option("channel", description="Channel to analyze", type=discord.TextChannel, required=True)
@discord.option("mode", description="Group by time", choices=["day", "month", "year"], default="day")
async def channelstats(ctx: discord.ApplicationContext, channel: discord.TextChannel, mode: str):
    await ctx.defer()

    channel_data = load_channel_data(ctx.guild.id, channel.id)
    if not channel_data["message_counts"]:
        await ctx.respond(f"No data available for {channel.name}")
        return

    total_messages = sum(channel_data["message_counts"].values())
    label, max_count = get_max_message_day(channel_data["message_counts"], mode)

    mean_val = compute_stat_from_counts(channel_data["message_counts"], mode, "mean")
    median_val = compute_stat_from_counts(channel_data["message_counts"], mode, "median")

    response = f"**Statistics for #{channel.name}:**\n"
    response += f"• Total messages: **{total_messages}**\n"
    response += f"• Most active {mode}: **{label}** ({max_count} messages)\n"
    if mean_val:
        response += f"• Mean messages per {mode}: **{mean_val}**\n"
    if median_val:
        response += f"• Median messages per {mode}: **{median_val}**"

    await ctx.respond(response)

@bot.slash_command(name="messagedistribution", description="Plot message frequency over time.")
@discord.option("time_unit", str, description="Group messages by: day, month, or year", choices=["day", "month", "year"])
@discord.option("bin_size", int, description="Number of time units per bin", default=1, min_value=1, max_value=365)
@discord.option("user", discord.User, description="(Optional) Analyze only this user's messages", required=False)
async def messagedistribution(ctx: discord.ApplicationContext, time_unit: str, bin_size: int, user: discord.User = None):
    await ctx.defer()

    now = datetime.now(timezone.utc)
    start_of_history = now - timedelta(days=365*2)  # Adjustable range

    guild_summary = load_guild_summary(ctx.guild.id)

    if user is None:
        data = guild_summary["message_counts"]
    else:
        data = guild_summary["user_message_counts"].get(str(user.id))

    x_vals, y_vals = prep_graph(data, time_unit, cumulative=False)

    message_counts = np.array(y_vals)

    hist, bin_edges = np.histogram(message_counts, bins=np.arange(0, message_counts.max() + bin_size, bin_size))

    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i + 1] - 1)}" for i in range(len(hist))]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(bin_labels, hist, color='skyblue')
    plt.xlabel(f'Message Counts ({time_unit}, bin size = {bin_size})')
    plt.ylabel(f'# of {time_unit}s')
    plt.title(f"Message Distribution {'for ' + user.name if user else 'for Server'}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    image_path = f"{ctx.guild.id}_distribution.png"
    plt.savefig(image_path, dpi=150, bbox_inches='tight')
    plt.close()

    await ctx.respond(file=discord.File(image_path))
    os.remove(image_path)

bot.run("token")