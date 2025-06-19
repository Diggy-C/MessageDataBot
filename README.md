# MessageDataBot
Analyze your server's message history over time! Collect data and plot cool graphs!

## How to use
Create a discord bot at the Discord Developer Portal: https://discord.com/developers/applications
Download the .py file, and at the bottom of the file, replace "Token" in bot.run("Token") at the bottom of the file with your bot's token. Keep the file running until you see it has finished scanning all the messages in the server (might take a while, usually approximately 90 min per 500k messages). Once it's finished, you can run all the commands! The message data will be stored in JSON's so it only takes a long time for the initial reading, all commands should be relatively quick after

## Commands
### Graph
Graph the message history of the server by day, month, or year. Can also be cumulative

### User Graph
Same as graph, but you put up to 5 users and see their message history on the same graph

### Message Distribution
Plots a histogram of message history. Can pick a specific user or total, and can pick bin size for histogram

### Emotions
Perform sentiment analysis, using NRCLex to determine the emotions from a certain number of messages from a selected user

### User Max
Find the max day from a user (may be one day ahead of reality)

### User Stat
Collect the mean or median messages per day, month, or year for a user

### Channel Stats
Show statistics for a specific channel 

### Compare Sentiment
Generate a graph showing the most positive/negative users in the past certain number of days

### Max Day
Determine the day the server was most active

### Stat
Gets mean/median day, month, or year for the entire server
