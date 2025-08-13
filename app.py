import streamlit as st
import preprocessor,helper
import matplotlib.pyplot as plt
import seaborn as sns

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)
    df = helper.add_sentiment(df)



    #fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")



    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):
        

        #stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics") 
        col1, col2, col3, col4 = st.columns(4)
            
        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        
        with col2:
            st.header("Total Words")
            st.title(words)

        with col3:
             st.header("Media Shared")
             st.title(num_media_messages)

        with col4:
             st.header("Links Shared")
             st.title(num_links)



        #Monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig,ax = plt.subplots() 
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        #daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig,ax = plt.subplots() 
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)


        #Activity map
        st.title("Activity Map")
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='blue')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        
        
        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user,df)
        fig,ax = plt.subplots()
        sns.heatmap(user_heatmap, ax=ax)
        st.pyplot(fig)


       #finding the busiest users in the group(Group level)

        if selected_user == 'Overall':
            st.title("Most Busy Users")
            x,new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            
            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color = 'red' )
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.dataframe(new_df)


        #wordcloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user,df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)


        #most common words
        # most_common_df = helper.most_common_words(selected_user,df)

        # fig,ax = plt.subplots()

        # ax.barh(most_common_df[0], most_common_df[1])
        # plt.xticks(rotation='vertical')
        # st.title("Most Common Words")
        # st.pyplot(fig)



        st.title("Most Common Words (Cleaned)")
        mcw = helper.most_common_words_clean(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(mcw['word'], mcw['count'])
        plt.gca().invert_yaxis()
        st.pyplot(fig)





        #emoji analysis
        emoji_df = helper.emoji_helper(selected_user,df)
        st.title("Emoji Analysis")

        col1,col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df[0])

        with col2:
            fig,ax = plt.subplots()
            ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")
            st.pyplot(fig)
        

        

        # Sentiment breakdown
        st.title("Sentiment Overview")
        sent_counts = helper.sentiment_breakdown(selected_user, df)
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(sent_counts)  # simple and quick
        with col2:
            st.write((sent_counts / sent_counts.sum() * 100).round(2).astype(str) + '%')

        # Sentiment over time
        st.title("Average Sentiment by Day")
        sent_timeline = helper.sentiment_daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(sent_timeline['only_date'], sent_timeline['compound'])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)



        st.title("Discovered Topics (LDA)")
        topics = helper.lda_topics(selected_user, df, n_topics=5, n_top_words=8)
        if not topics:
           st.info("Not enough text to extract topics.")
        else:
         for t in topics:
            st.write(f"**Topic {t['topic']}:** " + ", ".join(t['terms']))


        st.title("Conversation Dynamics")
        col1, col2 = st.columns(2)
        with col1:
         st.subheader("Who starts conversations?")
         starters = helper.conversation_starters(selected_user, df, gap_minutes=30)
         st.bar_chart(starters)

        with col2:
          st.subheader("Median response time (seconds)")
          rtimes = helper.median_response_time(df)
          st.dataframe(rtimes)

        summary_txt = helper.build_summary_text(selected_user, df, sent_counts, starters, rtimes)
        st.download_button("Download Summary (.txt)", data=summary_txt, file_name="chat_summary.txt")







        
     
