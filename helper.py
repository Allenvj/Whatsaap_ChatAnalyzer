from urlextract import URLExtract
from wordcloud import WordCloud
extract = URLExtract()
import pandas as pd
from collections import Counter
import emoji
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy as np


def fetch_stats(selected_user, df):

  if selected_user != 'Overall':
    df = df[df['user'] == selected_user]


  #fetch no of messages
  num_messages = df.shape[0]

  #fetch no of words
  words = []
  for message in df['message']:
      words.extend(message.split())


  
  #fetch no of media messages
  num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]


  #fetch no of links shared
  links = []
  for message in df['message']:
      links.extend(extract.find_urls(message))

  return num_messages, len(words), num_media_messages, len(links)

def most_busy_users(df):
   x = df['user'].value_counts().head()
   df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(columns={'user': 'name', 'count': 'percent'})

   return x, df


def create_wordcloud(selected_user,df):

   if selected_user != 'Overall':
    df = df[df['user'] == selected_user]

   temp = df[df['user'] != "group_notification"]
   temp = temp[temp['message'] != '<Media omitted>\n'] 

   wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
 
   df_wc = wc.generate(temp['message'].str.cat(sep=" "))
   return df_wc


# def most_common_words(selected_user, df):
   
#    if selected_user != 'Overall':
#     df = df[df['user'] == selected_user]

#    temp = df[df['user'] != "group_notification"]
#    temp = temp[temp['message'] != '<Media omitted>\n'] 

#    words = []

#    for message in temp['message']:
#     for word in message.lower().split():
#       words.append(word)

#    most_common_df = pd.DataFrame(Counter(words).most_common(20))
#    return most_common_df
   

_URL_RE = re.compile(r'https?://\S+|www\.\S+')
_PUNCT_TABLE = str.maketrans('', '', string.punctuation)
_STOP = set(ENGLISH_STOP_WORDS) | {'media', 'omitted'}

def clean_tokenize(text):
    text = text.lower()
    text = _URL_RE.sub(' ', text)
    text = text.translate(_PUNCT_TABLE)
    toks = [t for t in text.split() if t and t not in _STOP]
    return toks


def most_common_words_clean(selected_user, df, topn=20):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[(df['user'] != 'group_notification') & (~df['message'].isin(['<Media omitted>', '<Media omitted>\n']))]
    from collections import Counter
    bag = Counter()
    for m in temp['message']:
        bag.update(clean_tokenize(m))
    return pd.DataFrame(bag.most_common(topn), columns=['word', 'count'])



def emoji_helper(selected_user,df):
  if selected_user != 'Overall':
    df = df[df['user'] == selected_user]

  emojis = []
  for message in df['message']:
    emojis.extend([e['emoji'] for e in emoji.emoji_list(message)])

  emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

  return emoji_df
 

def monthly_timeline(selected_user,df):

  if selected_user != 'Overall':
    df = df[df['user'] == selected_user]

  timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

  time = []
  for i in range(timeline.shape[0]):
    year = timeline['year'][i]
    month = timeline['month'][i]
    time.append(str(month) + "-" + str(year))

  timeline['time'] = time    

  return timeline


def daily_timeline(selected_user,df):

  if selected_user != 'Overall':
    df = df[df['user'] == selected_user]

  daily_timeline = df.groupby('only_date').count()['message'].reset_index()

  return daily_timeline


def week_activity_map(selected_user,df):

  if selected_user != 'Overall':
    df = df[df['user'] == selected_user]

  return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

  if selected_user != 'Overall':
    df = df[df['user'] == selected_user]

  return df['month'].value_counts()


def activity_heatmap(selected_user,df):

  if selected_user != 'Overall':
    df = df[df['user'] == selected_user]

  user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

  return user_heatmap
  

      

# Download once at runtime (safe to keep)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

_sia = SentimentIntensityAnalyzer()

def add_sentiment(df):
    # returns a new DataFrame with 'compound' and 'sentiment' columns
    out = df.copy()
    # ignore system messages
    mask = out['user'] != 'group_notification'
    scores = out.loc[mask, 'message'].apply(_sia.polarity_scores)
    out.loc[mask, 'compound'] = scores.apply(lambda d: d['compound'])
    def label(c):
        if c >= 0.05: return 'Positive'
        if c <= -0.05: return 'Negative'
        return 'Neutral'
    out.loc[mask, 'sentiment'] = out.loc[mask, 'compound'].apply(label)
    out['compound'] = out['compound'].fillna(0.0)
    out['sentiment'] = out['sentiment'].fillna('Neutral')
    return out

def sentiment_breakdown(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    counts = df['sentiment'].value_counts()
    return counts

def sentiment_daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    # average compound by date
    daily = df.groupby('only_date')['compound'].mean().reset_index()
    return daily










def lda_topics(selected_user, df, n_topics=5, n_top_words=8):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    docs = []
    for m in df[(df['user'] != 'group_notification')]['message']:
        toks = clean_tokenize(m)
        if toks:
            docs.append(' '.join(toks))
    if len(docs) < 10:
        return []  # not enough text

    vectorizer = CountVectorizer(max_df=0.8, min_df=3)
    X = vectorizer.fit_transform(docs)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method='batch')
    lda.fit(X)

    terms = vectorizer.get_feature_names_out()
    topics = []
    for idx, comp in enumerate(lda.components_):
        top_terms = [terms[i] for i in comp.argsort()[-n_top_words:][::-1]]
        topics.append({"topic": idx+1, "terms": top_terms})
    return topics


#first message after a long silence)
def conversation_starters(selected_user, df, gap_minutes=30):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    d = df[df['user'] != 'group_notification'].sort_values('date')
    d['prev_date'] = d['date'].shift(1)
    d['gap_min'] = (d['date'] - d['prev_date']).dt.total_seconds().div(60)
    starters = d[d['gap_min'] >= gap_minutes]
    return starters['user'].value_counts()

#median time a user takes to reply
    
def median_response_time(df):
    d = df[df['user'] != 'group_notification'].sort_values('date')
    d['prev_user'] = d['user'].shift(1)
    d['prev_date'] = d['date'].shift(1)
    # only replies when the user changes
    replies = d[d['user'] != d['prev_user']].copy()
    replies['resp_sec'] = (replies['date'] - replies['prev_date']).dt.total_seconds()
    return replies.groupby('user')['resp_sec'].median().sort_values()

def build_summary_text(selected_user, df, sent_counts, starters, rtimes):
    lines = []
    lines.append(f"WhatsApp Chat Summary for: {selected_user}")
    lines.append(f"Total messages: {df.shape[0]}")
    if 'sentiment' in df.columns:
        lines.append("Sentiment breakdown:")
        for s, c in sent_counts.items():
            lines.append(f"  - {s}: {c}")
    lines.append("\nTop conversation starters:")
    for u, c in starters.items():
        lines.append(f"  - {u}: {c}")
    lines.append("\nMedian response times (seconds):")
    for u, s in rtimes.items():
        lines.append(f"  - {u}: {int(s)}")
    return "\n".join(lines)