from typing import Dict, Text
import csv

import pandas as pd

import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

from tensorflow.keras import Model
from tensorflow.keras.models import Model

RUN_DIR_HOME='/Users/thomaslorenc/Sites/flag/tensor/'
RUN_DIR_EC2='/home/ec2-user/tensor/'
RUN_DIR='/Users/thomaslorenc/Sites/flag/tensor/'


#######################################################################################################################################
# STEP 1 - DATA PREP - TITLE vs STYLE ACC 
#######################################################################################################################################

def tf_prep_propensity_score():


    my_style_prop_lol = [['style_Sweater_Hoodie', ['Hoodie']], ['style_Dress_Shirt', ['Dress Shirt','SSBD','Custom Shirt']], 
    ['style_Casual_Shirt', ['Shirt','Polo','Tech Polo','Casual Button Down','Henley','Long Sleeve Polo','V-Neck','Tech Crewneck']], 
    ['style_Suit', ['Jacket', 'Suit']],['style_Tuxedo', ['Tuxedo',  'Tuxedo Pants' ,'Tuxedo Jacket']],['style_Outerwear', ['Overcoat',  'Sweater']],
    ['style_Tech_Pant', ['Tech Pant']],['style_Shorts', ['Shorts','Short']]]

    prod_file_in2 = '/home/ec2-user/tensor/sales_SL_ALL_IN.csv'
    sales_cust_file_in_df = pd.read_csv(prod_file_in2)

    sales_parquet_file_name = '/home/ec2-user/tensor/recs_train_groups.parquet'
    ssales_parquet_file_name_df = pd.read_parquet(sales_parquet_file_name)
    print(ssales_parquet_file_name_df.columns)

    my_style_cats_recs_train = ['Shirt', 'Polo' , 'Long Sleeve Polo' , 'Overcoat' ,'Jacket', 'Tech Polo' ,'Dress Shirt' ,'SSBD' ,'Casual Button Down', 
    'Suit', 'Henley', 'Pants' ,'Sweater', 'Tuxedo' ,'Custom Shirt', 'Shorts' ,'Short' ,'V-Neck', 'Hoodie' ,'Crewneck' ,'Tech Pant' ,'Tech Crewneck','Tuxedo Pants' ,'Tuxedo Jacket']

    my_title_category = ssales_parquet_file_name_df['title_category'].unique()
    print(my_title_category)

    prod_file_in = '/home/ec2-user/tensor/prop_scores_by_style_without_NaNs.csv'
    prod_file_in_df = pd.read_csv(prod_file_in)
    my_prop_scores_by_style_without_NaNs = ['style_Accessory', 'style_Sweater_Hoodie', 'style_Dress_Shirt',
       'style_Casual_Shirt', 'style_Suit', 'style_Tuxedo', 'style_Outerwear',
       'style_Tech_Pant', 'style_Shorts']



    my_style_lol = []
    for mys in my_style_cats_recs_train:
        mys_list = []
        my_product_ids = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['title_category'] == mys]['product_id']
        product_id = 0
        for my_product_ids_next in my_product_ids:
            mys_list.append(my_product_ids_next)
        mys_list_set = set(mys_list)
        myr = [mys,mys_list_set]
        my_style_lol.append(myr)


    sales_file_in = '/home/ec2-user/tensor/sales_style_color_price_v8.csv'
    sales_file_in_df = pd.read_csv(sales_file_in)

   

    product_file_in = '/home/ec2-user/tensor/product_style_color_price_v8.csv'
    product_file_in_df = pd.read_csv(product_file_in)


    my_h = ['userId', 'movieId', 'rating','timestamp']
    my_lol_out = []

    my_lol_out.append(my_h)

    k = 0 
    # 1,christopherjroth@me.com,383528296,0,0,0,0,0,1

    prod_file_in_props = '/home/ec2-user/tensor/prop_scores_by_style_without_NaNs.csv'
    prod_file_in_props_df = pd.read_csv(prod_file_in_props)

    prod_clicks_file_in_props = '/home/ec2-user/tensor/email_clicks_for_tom.csv'
    prod_clicks_file_in_props_df = pd.read_csv(prod_file_in_props)



    mean_style_Dress_Shirt = prod_clicks_file_in_props_df["style_Dress_Shirt"].mean()
    print(mean_style_Dress_Shirt)

    mean_style_Sweater_Hoodiec = prod_clicks_file_in_props_df["style_Sweater_Hoodie"].mean()

    mean_style_Casual_Shirtc = prod_clicks_file_in_props_df["style_Casual_Shirt"].mean()

    mean_style_Casual_Shirtc = prod_clicks_file_in_props_df["style_Suit"].mean()

    mean_style_Tuxedoc = prod_clicks_file_in_props_df["style_Tuxedo"].mean()

    mean_style_Outerwearc = prod_clicks_file_in_props_df["style_Outerwear"].mean()

    mean_style_Tech_Pantc = prod_clicks_file_in_props_df["style_Tech_Pant"].mean()

    mean_style_Shorts= prod_clicks_file_in_props_df["style_Shorts"].mean()


    mean_style_Dress_Shirt = prod_file_in_props_df["style_Dress_Shirt"].mean()
    print(mean_style_Dress_Shirt)

    mean_style_Sweater_Hoodie = prod_file_in_props_df["style_Sweater_Hoodie"].mean()

    mean_style_Casual_Shirt = prod_file_in_props_df["style_Casual_Shirt"].mean()

    mean_style_Suit = prod_file_in_props_df["style_Suit"].mean()

    mean_style_Tuxedo = prod_file_in_props_df["style_Tuxedo"].mean()

    mean_style_Outerwear = prod_file_in_props_df["style_Outerwear"].mean()

    mean_style_Tech_Pant = prod_file_in_props_df["style_Tech_Pant"].mean()

    mean_style_Shorts= prod_file_in_props_df["style_Shorts"].mean()
    my_mean_lol =[['style_Dress_Shirt',mean_style_Dress_Shirt],['style_Sweater_Hoodie',mean_style_Sweater_Hoodie],['style_Casual_Shirt',mean_style_Casual_Shirt],['style_Suit',mean_style_Suit],['style_Tuxedo',mean_style_Tuxedo],['style_Outerwear',mean_style_Outerwear],['style_Tech_Pant',mean_style_Tech_Pant],['style_Shorts',mean_style_Shorts]]




    for index, row in sales_file_in_df.iterrows():
        act_customer_id = row["userId"]
        act_prod_id = row["movieId"]
        act_rating = row["rating"]
        act_timestamp = row["timestamp"]
        print('act_prod_id')
        print(act_prod_id)
        print('act_customer_id')

        print(act_customer_id)
        customer_emails = sales_cust_file_in_df.loc[sales_cust_file_in_df['customer_id'] == act_customer_id]['customer_email']
        customer_email = 'Shirt'
        for customer_emails_next in customer_emails:
            customer_email = customer_emails_next
            break


        if customer_email != 'Shirt':

            for nextr in my_style_lol:
                my_style_maybe = nextr[0]
                my_prods_maybe = nextr[1]
                if act_prod_id in my_prods_maybe:
                    my_style = my_style_maybe
                    break


            for nextr in my_style_prop_lol:
                my_prop_style_maybe = nextr[0]
                my_style_maybe = nextr[1]
                if my_style in my_style_maybe:
                    my_style_prop = my_prop_style_maybe
                    break


            print(my_style_prop)
            print(customer_email)

            my_style_prop_scores = prod_file_in_props_df.loc[prod_file_in_props_df['email'] == customer_email][my_style_prop]
            my_style_prop_score = 'Shirt'
            for my_style_prop_scores_next in my_style_prop_scores:
                my_style_prop_score = my_style_prop_scores_next
                break


            if my_style_prop_score != 'Shirt':

                
                for nextr in my_mean_lol:
                    my_prop_name_maybe = nextr[0]
                    my_style_prop_score_mean_maybe = nextr[1]
                    if my_prop_name_maybe == my_style_prop:
                        my_style_prop_score_mean = my_style_prop_score_mean_maybe


                print(my_style_prop_score)
                print(my_style_prop_score_mean)

                if my_style_prop_score >= my_style_prop_score_mean:
                    print('hit')
                    if act_rating == 1 : act_rating = 3 
                    elif act_rating == 2 : act_rating = 4
                myr = [act_customer_id, act_prod_id,act_rating,act_timestamp]
                my_lol_out.append(myr)

            else: 
                myr = [act_customer_id, act_prod_id,act_rating,act_timestamp]
                my_lol_out.append(myr)

        else:
            myr = [act_customer_id, act_prod_id,act_rating,act_timestamp]
            my_lol_out.append(myr)

    prop_out_data = '/home/ec2-user/tensor/sales_style_color_price_quint_2_19.csv'

    with open(prop_out_data, "w") as f:
        writer = csv.writer(f)
        writer.writerows(my_lol_out)



def tf_data_prep():
    tf_prep_propensity_score()

    my_style_cats_recs_train = ['Shirt', 'Polo' , 'Long Sleeve Polo' , 'Overcoat' ,'Jacket', 'Tech Polo' ,'Dress Shirt' ,'SSBD' ,'Casual Button Down', 
    'Suit', 'Henley', 'Pants' ,'Sweater', 'Tuxedo' ,'Custom Shirt', 'Shorts' ,'Short' ,'V-Neck', 'Hoodie' ,'Crewneck' ,'Tech Pant' ,'Tech Crewneck','Tuxedo Pants' ,'Tuxedo Jacket']
    import time
    import datetime
    prod_file_in = RUN_DIR_EC2 + 'products_export_1.csv'
    prod_file_in_df = pd.read_csv(prod_file_in)

    sales_parquet_file_name = RUN_DIR_EC2 + 'recs_train.parquet'

    ssales_parquet_file_name_df = pd.read_parquet(sales_parquet_file_name)
    print(ssales_parquet_file_name_df.columns)
    ssales_parquet_file_name_df = ssales_parquet_file_name_df[
        ssales_parquet_file_name_df.product_id != 0]
    ssales_parquet_file_name_df = ssales_parquet_file_name_df[
        ssales_parquet_file_name_df.order_id != 0]
    ssales_parquet_file_name_df = ssales_parquet_file_name_df[
        ssales_parquet_file_name_df.total_sales != 0]


    my_prods = ssales_parquet_file_name_df['product_id'].unique()
    my_lol_out = []
    my_h = ['movie_title','movieId', 'title', 'genres']
    my_lol_out.append(my_h)

    my_good_prods = []
    my_lol_out2 = []
    my_h2 = ['user_id', 'movie_title','movieId', 'rating', 'timestamp']
    my_lol_out2.append(my_h2)

    for next_prod in my_prods:
        my_product_titles = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['product_id'] == next_prod]['product_title']
        my_prod_title = 'Shirt'
        for my_product_titles_next in my_product_titles:
            my_prod_title = my_product_titles_next
            break
        mptc = my_prod_title.replace('"', '')
        results = f"'{mptc}'"

        my_product_types = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['product_title'] == my_prod_title]['title_category']
        my_product_type = 'Shirt'
        for my_product_type_next in my_product_types:
            my_product_type = my_product_type_next
            break

        ## if not in then contin eu 

        if my_product_type in my_style_cats_recs_train:

                my_product_colors = prod_file_in_df.loc[prod_file_in_df['Title'] == my_prod_title]['Option2 Value']
                my_prod_color = 'Black'
                for my_product_colors_next in my_product_colors:
                    my_prod_color = my_product_colors_next
                    break

                my_product_total_sales = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['product_id'] == next_prod]['total_sales']
                my_product_total_sale = 0
                for my_product_total_sales_next in my_product_total_sales:
                    my_product_total_sale = my_product_total_sales_next
                    break
                my_product_total_sale_bin = 1
                if  my_product_total_sale == my_product_total_sale :
                    if my_product_total_sale < 50 : my_product_total_sale_bin = 1
                    elif 50 < my_product_total_sale and my_product_total_sale < 100 :  my_product_total_sale_bin = 2
                    elif 100 < my_product_total_sale and my_product_total_sale < 200 :  my_product_total_sale_bin = 3
                    elif 200 < my_product_total_sale and my_product_total_sale < 300 :  my_product_total_sale_bin = 4
                    else:        
                        my_product_total_sale_bin = 5



                if my_prod_title == my_prod_title and my_prod_color == my_prod_color and my_prod_color != 'Black' and my_product_type == my_product_type and my_product_total_sale != 0:
                    mpc = my_prod_color.replace(" ", "")
                    my_prod_feature = my_product_type + '|' + mpc + '|' + str(my_product_total_sale_bin)
                    myrow = [results,next_prod, next_prod, my_prod_feature]
                    my_lol_out.append(myrow)
                    my_good_prods.append(next_prod)


    pred_file_in_cat_codes_out = RUN_DIR_EC2 + 'product_style_color_price_1_19.csv'

    with open(pred_file_in_cat_codes_out, "w") as f:
        writer = csv.writer(f)
        writer.writerows(my_lol_out)




    for index, row in ssales_parquet_file_name_df.iterrows():
        next_cust = row["customer_id"]
        ncs = str(next_cust)
        next_cust_results = f"'{ncs}'"
        my_prod = row["product_id"]
        my_prod_title = row["product_title"]
        mptc = my_prod_title.replace('"', '')
        result = f"'{mptc}'"
        if my_prod in my_good_prods: 
            my_net_quantity = row["net_quantity"]
            my_date = str(row["hour"])
            mydt = time.mktime(datetime.datetime.strptime(my_date, "%Y-%m-%d %H:%M:%S").timetuple())
            dec = str(mydt)
            dec_2 = dec[:-2]
            myrow = [next_cust_results, result,my_prod, my_net_quantity, dec_2]
            my_lol_out2.append(myrow)

    pred_file_in_cat_codes_out2 = RUN_DIR_EC2 + 'sales_style_color_price_quint_2_19.csv'


    with open(pred_file_in_cat_codes_out2, "w") as f:
        writer = csv.writer(f)
        writer.writerows(my_lol_out2)




#######################################################################################################################################
# STEP 2  - RETREVAL MODEL 
#######################################################################################################################################


def tf_recs_ret_deep_2():
    DATA_URL_product = '/home/ec2-user/tensor/sales_style_color_price_quint_2_19.csv'
    DATA_URL_product_df = pd.read_csv(DATA_URL_product)


    DATA_URL_product2 = '/home/ec2-user/tensor/product_style_color_price_2_15.csv'
    DATA_URL_product_df2 = pd.read_csv(DATA_URL_product2)


    ratings = tf.data.Dataset.from_tensor_slices(dict(DATA_URL_product_df)).map(lambda x: { "movie_title": x["movie_title"], "user_id": x["user_id"],"user_rating": x["rating"],"timestamp": x["timestamp"]  })
    movies = tf.data.Dataset.from_tensor_slices(dict(DATA_URL_product_df2)).map(lambda x: { "movie_title": x["movie_title"], "style": x["genres"]  })


    tf.random.set_seed(42)
    shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)


    #######################################################################################################################################
    # User_id title time stamp stylke vocabulary and embedding model  
    #######################################################################################################################################

    #movie_titles = movies.batch(1_000)k
    movie_titles = movies.batch(1_000_000).map(lambda x: x["movie_title"])

    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    unique_movie_titles[:14]


    style_color_price_text = tf.keras.layers.TextVectorization()
    style_color_price_text.adapt(movies.map(lambda x: x["style"]))


    timestamp_normalization = tf.keras.layers.Normalization(axis=None)
    timestamp_normalization.adapt(ratings.map(lambda x: x["timestamp"]).batch(1024))

    max_timestamp = ratings.map(lambda x: x["timestamp"]).reduce(tf.cast(0, tf.int64), tf.maximum).numpy().max()
    min_timestamp = ratings.map(lambda x: x["timestamp"]).reduce(np.int64(1e9), tf.minimum).numpy().min()
    timestamp_buckets = np.linspace(min_timestamp, max_timestamp, num=1000)

    timestamp_embedding_model = tf.keras.Sequential([tf.keras.layers.Discretization(timestamp_buckets.tolist()),
        tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32)
        ])


    user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    user_ids_vocabulary.adapt(ratings.map(lambda x: x["user_id"]))

    movie_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    movie_titles_vocabulary.adapt(movies)

    movie_titles = ratings.batch(1_000_000).map(lambda x: x["movie_title"])
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    embedding_dimension = 32


    timestamp_normalization = tf.keras.layers.Normalization(axis=None)
    timestamp_normalization.adapt(ratings.map(lambda x: x["timestamp"]).batch(1024))


    class UserModel(tf.keras.Model):

      def __init__(self):
        super().__init__()

        self.user_embedding = tf.keras.Sequential([
            user_id_lookup,
            tf.keras.layers.Embedding(user_id_lookup.vocab_size(), 32),
        ])
        self.timestamp_embedding = tf.keras.Sequential([
          tf.keras.layers.Discretization(timestamp_buckets.tolist()),
          tf.keras.layers.Embedding(len(timestamp_buckets) + 2, 32)
        ])
        self.normalized_timestamp = tf.keras.layers.Normalization(
            axis=None
        )

      def call(self, inputs):

        # Take the input dictionary, pass it through each input layer,
        # and concatenate the result.
        return tf.concat([
            self.user_embedding(inputs["user_id"]),
            self.timestamp_embedding(inputs["timestamp"]),
            tf.reshape(self.normalized_timestamp(inputs["timestamp"]), (-1, 1))
        ], axis=1)


    user_model = UserModel()
    user_model.normalized_timestamp.adapt(ratings.map(lambda x: x["timestamp"]).batch(128))

    class MovieModel(tf.keras.Model):

      def __init__(self):
        super().__init__()

        max_tokens = 10_000

        self.title_embedding = tf.keras.Sequential([
          movie_title_lookup,
          tf.keras.layers.Embedding(movie_title_lookup.vocab_size(), 32)
        ])
        self.style_text_embedding = tf.keras.Sequential([
          tf.keras.layers.TextVectorization(max_tokens=max_tokens),
          tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
          # We average the embedding of individual words to get one embedding vector
          # per title.
          tf.keras.layers.GlobalAveragePooling1D(),
        ])

      def call(self, inputs):
        return tf.concat([
            self.title_embedding(inputs["movie_title"]),
            self.style_text_embedding(inputs["style"]),
        ], axis=1)

    movie_model = MovieModel()

    movie_model.style_text_embedding.layers[0].adapt(movies.map(lambda x: x["style"]))

    # Define your objectives.
    task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
        movies.batch(128).map(movie_model)
    )
    )
    movie_titles = ratings.batch(1_000_000).map(lambda x: x["movie_title"])
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))


    class MovielensModel(tfrs.Model):

        def __init__(self, user_model, movie_model):
            super().__init__()
            self.movie_model: tf.keras.Model = movie_model
            self.user_model: tf.keras.Model = user_model
            self.task: tf.keras.layers.Layer = task

        def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
            # We pick out the user features and pass them into the user model.
            user_embeddings = self.user_model(features["user_id"])
            # And pick out the movie features and pass them into the movie model,
            # getting embeddings back.
            positive_movie_embeddings = self.movie_model(features["movie_title"])

            return self.task(user_embeddings, positive_movie_embeddings)

    # Create a retrieval model.
    model = MovielensModel(user_model, movie_model)
    #model = MovielensModel(user_model, movie_model)
    #model = MovielensModel()
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

    # Train for 3 epochs.
    model.fit(ratings.batch(4096), epochs=10)

    # Use brute-force search to set up retrieval using the trained representations.
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    index.index_from_dataset(
        movies.batch(100).map(lambda title: (title, model.movie_model(title))))

    # Get some recommendations.
    _, titles = index(np.array(["42"]))
    print(f"Top 3 recommendations for user 42: {titles[0, :6]}")



    thing, titles = index(np.array([user_ids_vocabulary.get_vocabulary()[1]]))
    print(f"Top 10 recommendations for user {user_ids_vocabulary.get_vocabulary()[1]}: {titles}")
    print(f"thing: {thing}")
    thing, titles = index(np.array([user_ids_vocabulary.get_vocabulary()[2]]))
    print(f"Top 10 recommendations for user {user_ids_vocabulary.get_vocabulary()[2]}: {titles}")
    print(f"thing: {thing}")
    thing, titles = index(np.array([user_ids_vocabulary.get_vocabulary()[3]]))
    print(f"Top 10 recommendations for user {user_ids_vocabulary.get_vocabulary()[3]}: {titles}")
    print(f"thing: {thing}")
    sales_parquet_file_name = RUN_DIR_EC2 + 'product_style_color_price_v100.csv'

    ssales_parquet_file_name_df = pd.read_csv(sales_parquet_file_name)



    my_pred_list_lists_out_end_title = []
    my_ht = ['customer_id', 'top product_title 1', 'top product_title 2',
            'top product_title 3', 'top product_title 4', 'top product_title 5', 'top product_title 6',
            'top product_title 7', 'top product_title 8', 'top product_title 9', 'top product_title 10']
    my_pred_list_lists_out_end_title.append(my_ht)
    my_pred_list_lists_out_end = []
    my_h = ['customer_id', 'top product_id 1', 'top product_id 2',
            'top product_id 3', 'top product_id 4', 'top product_id 5', 'top product_id 6',
            'top product_id 7', 'top product_id 8', 'top product_id 9', 'top product_id 10']
    my_pred_list_lists_out_end.append(my_h)

    k = 0 
    for next_user in unique_user_ids:
        k = k +1
        if k > 9: break
        nus = str(next_user)
        print(nus)
        mycid2 = user_ids_vocabulary.get_vocabulary()[k]
        uus2 = mycid2.replace('"', '')
        uus3 = uus2[1:]
        x4 = uus3[:-1]
        #_, titles = index(np.array([nus]))
        thing, titles = index(np.array([user_ids_vocabulary.get_vocabulary()[k]]))
        #print(f"Top 3 recommendations for user : {titles[0, :6]}")

        t_a = titles.numpy()
        y = 0
        #for x in range(1,len(t_a)):
        #output.write(str(data[k]))
        for q in t_a:
            for  xc in q :
                print('x')
                print(xc)
                xd = xc.decode("utf-8")
                x = xd.replace('"', '')
                print(x)
                y = y + 1
                if y == 1:
                    my_product_title_1  = x
                    my_product_id_0s = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['movie_title'] == x]['movieId']
                    my_product_id_0 = 'Shirt'
                    for my_product_id_0s_next in my_product_id_0s:
                        my_product_id_0 = my_product_id_0s_next
                        break
                    print(my_product_id_0)
                elif y == 2:
                    my_product_title_2  = x
                    my_product_id_1s = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['movie_title'] == x]['movieId']
                    my_product_id_1 = 'Shirt'
                    for my_product_id_1s_next in my_product_id_1s:
                        my_product_id_1 = my_product_id_1s_next
                        break
                    print(my_product_id_1)
                elif y == 3:
                    my_product_title_3  = x
                    my_product_id_2s = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['movie_title'] == x]['movieId']
                    my_product_id_2 = 'Shirt'
                    for my_product_id_2s_next in my_product_id_2s:
                        my_product_id_2 = my_product_id_2s_next
                        break
                    print(my_product_id_2)
                elif y == 4:
                    my_product_title_4  = x
                    my_product_id_3s = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['movie_title'] == x]['movieId']
                    my_product_id_3 = 'Shirt'
                    for my_product_id_3s_next in my_product_id_3s:
                        my_product_id_3 = my_product_id_3s_next
                        break
                    print(my_product_id_3)
                elif y == 5:
                    my_product_title_5  = x
                    my_product_id_4s = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['movie_title'] == x]['movieId']
                    my_product_id_4 = 'Shirt'
                    for my_product_id_4s_next in my_product_id_4s:
                        my_product_id_4 = my_product_id_4s_next
                        break
                    print(my_product_id_4)
                elif y == 6:
                    my_product_title_6  = x
                    my_product_id_5s = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['movie_title'] == x]['movieId']
                    my_product_id_5 = 'Shirt'
                    for my_product_id_5s_next in my_product_id_5s:
                        my_product_id_5 = my_product_id_5s_next
                        break
                    print(my_product_id_5)

                elif y == 7:
                    my_product_title_7  = x
                    my_product_id_6s = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['movie_title'] == x]['movieId']
                    my_product_id_6 = 'Shirt'
                    for my_product_id_6s_next in my_product_id_6s:
                        my_product_id_6 = my_product_id_6s_next
                        break
                    print(my_product_id_6)

                elif y == 8:
                    my_product_title_8  = x
                    my_product_id_7s = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['movie_title'] == x]['movieId']
                    my_product_id_7 = 'Shirt'
                    for my_product_id_7s_next in my_product_id_7s:
                        my_product_id_7 = my_product_id_7s_next
                        break
                    print(my_product_id_7)


                elif y == 9:
                    my_product_title_9  = x
                    my_product_id_8s = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['movie_title'] == x]['movieId']
                    my_product_id_8 = 'Shirt'
                    for my_product_id_8s_next in my_product_id_8s:
                        my_product_id_8 = my_product_id_8s_next
                        break
                    print(my_product_id_8)


                elif y == 10:
                    my_product_title_10  = x
                    my_product_id_9s = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['movie_title'] == x]['movieId']
                    my_product_id_9 = 'Shirt'
                    for my_product_id_9s_next in my_product_id_9s:
                        my_product_id_9 = my_product_id_9s_next
                        break
                    print(my_product_id_9)



                print(x)
            nudd = next_user.decode("utf-8")
            uus = nudd.replace('"', '')
            uus1 = uus[1:]
            uus2 = uus1[:-1]
            my_next_row = [x4,my_product_id_0,my_product_id_1,my_product_id_2,my_product_id_3,my_product_id_4, my_product_id_5, my_product_id_6,my_product_id_7,my_product_id_8,my_product_id_9]
            
            my_next_rowt = [x4,my_product_title_1,my_product_title_2,my_product_title_3,my_product_title_4,my_product_title_5, my_product_title_6, my_product_title_7,my_product_title_8,my_product_title_9,my_product_title_10]
            my_pred_list_lists_out_end.append(my_next_row)
            my_pred_list_lists_out_end_title.append(my_next_rowt)

    pred_file_out = '/home/ec2-user/tensor/tf_recs_2_17_v1.csv'
    with open(pred_file_out, "w") as f:
        writer = csv.writer(f)
        writer.writerows(my_pred_list_lists_out_end)
    pred_file_out = '/home/ec2-user/tensor/tf_recs_2_17_v1_TITLE.csv'
    with open(pred_file_out, "w") as f:
        writer = csv.writer(f)
        writer.writerows(my_pred_list_lists_out_end_title)



def tf_recs_ret_deep():
    DATA_URL_product = '/home/ec2-user/tensor/sales_style_color_price_quint_1_19.csv'
    DATA_URL_product_df = pd.read_csv(DATA_URL_product)


    DATA_URL_product2 = '/home/ec2-user/tensor/product_style_color_price_1_19.csv'
    DATA_URL_product_df2 = pd.read_csv(DATA_URL_product2)


    ratings = tf.data.Dataset.from_tensor_slices(dict(DATA_URL_product_df)).map(lambda x: { "movie_title": x["movie_title"], "user_id": x["user_id"],"user_rating": x["rating"] })
    movies = tf.data.Dataset.from_tensor_slices(dict(DATA_URL_product_df2)).map(lambda x: x["movie_title"])
    # Select the basic features.
    #ratings = ratings.map(lambda x: { "movie_title": x["movie_title"], "user_id": x["user_id"] })
    #movies = movies.map(lambda x: x["movie_title"])
    tf.random.set_seed(42)
    shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)

    movie_titles = movies.batch(1_000)
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    unique_movie_titles[:14]

    user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    user_ids_vocabulary.adapt(ratings.map(lambda x: x["user_id"]))

    movie_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    movie_titles_vocabulary.adapt(movies)

    movie_titles = ratings.batch(1_000_000).map(lambda x: x["movie_title"])
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    embedding_dimension = 32


    # Define user and movie models.
    user_model = tf.keras.Sequential([
        user_ids_vocabulary,
        tf.keras.layers.Embedding(user_ids_vocabulary.vocab_size(), 64)
    ])
    movie_model = tf.keras.Sequential([
        movie_titles_vocabulary,
        tf.keras.layers.Embedding(movie_titles_vocabulary.vocab_size(), 64)
    ])

    # Define your objectives.
    task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
        movies.batch(128).map(movie_model)
    )
    )
    movie_titles = ratings.batch(1_000_000).map(lambda x: x["movie_title"])
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))


    class MovielensModel(tfrs.Model):

        def __init__(self, user_model, movie_model):
            super().__init__()
            self.movie_model: tf.keras.Model = movie_model
            self.user_model: tf.keras.Model = user_model
            self.task: tf.keras.layers.Layer = task

        def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
            # We pick out the user features and pass them into the user model.
            user_embeddings = self.user_model(features["user_id"])
            # And pick out the movie features and pass them into the movie model,
            # getting embeddings back.
            positive_movie_embeddings = self.movie_model(features["movie_title"])

            return self.task(user_embeddings, positive_movie_embeddings)

    # Create a retrieval model.
    model = MovielensModel(user_model, movie_model)
    #model = MovielensModel(user_model, movie_model)
    #model = MovielensModel()
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

    # Train for 3 epochs.
    model.fit(ratings.batch(4096), epochs=10)

    # Use brute-force search to set up retrieval using the trained representations.
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    index.index_from_dataset(
        movies.batch(100).map(lambda title: (title, model.movie_model(title))))

    # Get some recommendations.
    _, titles = index(np.array(["42"]))
    print(f"Top 3 recommendations for user 42: {titles[0, :6]}")



    thing, titles = index(np.array([user_ids_vocabulary.get_vocabulary()[1]]))
    print(f"Top 10 recommendations for user {user_ids_vocabulary.get_vocabulary()[1]}: {titles}")
    print(f"thing: {thing}")
    thing, titles = index(np.array([user_ids_vocabulary.get_vocabulary()[2]]))
    print(f"Top 10 recommendations for user {user_ids_vocabulary.get_vocabulary()[2]}: {titles}")
    print(f"thing: {thing}")
    thing, titles = index(np.array([user_ids_vocabulary.get_vocabulary()[3]]))
    print(f"Top 10 recommendations for user {user_ids_vocabulary.get_vocabulary()[3]}: {titles}")
    print(f"thing: {thing}")
    sales_parquet_file_name = RUN_DIR_EC2 + 'product_style_color_price_v100.csv'

    ssales_parquet_file_name_df = pd.read_csv(sales_parquet_file_name)



    my_pred_list_lists_out_end_title = []
    my_ht = ['customer_id', 'top product_title 1', 'top product_title 2',
            'top product_title 3', 'top product_title 4', 'top product_title 5', 'top product_title 6',
            'top product_title 7', 'top product_title 8', 'top product_title 9', 'top product_title 10']
    my_pred_list_lists_out_end_title.append(my_ht)
    my_pred_list_lists_out_end = []
    my_h = ['customer_id', 'top product_id 1', 'top product_id 2',
            'top product_id 3', 'top product_id 4', 'top product_id 5', 'top product_id 6',
            'top product_id 7', 'top product_id 8', 'top product_id 9', 'top product_id 10']
    my_pred_list_lists_out_end.append(my_h)

    k = 0 
    for next_user in unique_user_ids:
        k = k +1
        if k > 90000: break
        nus = str(next_user)
        print(nus)
        mycid2 = user_ids_vocabulary.get_vocabulary()[k]
        uus2 = mycid2.replace('"', '')
        uus3 = uus2[1:]
        x4 = uus3[:-1]
        #_, titles = index(np.array([nus]))
        thing, titles = index(np.array([user_ids_vocabulary.get_vocabulary()[k]]))
        #print(f"Top 3 recommendations for user : {titles[0, :6]}")

        t_a = titles.numpy()
        y = 0
        #for x in range(1,len(t_a)):
        #output.write(str(data[k]))
        for q in t_a:
            for  xc in q :
                print('x')
                print(xc)
                xd = xc.decode("utf-8")
                x = xd.replace('"', '')
                print(x)
                y = y + 1
                if y == 1:
                    my_product_title_1  = x
                    my_product_id_0s = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['movie_title'] == x]['movieId']
                    my_product_id_0 = 'Shirt'
                    for my_product_id_0s_next in my_product_id_0s:
                        my_product_id_0 = my_product_id_0s_next
                        break
                    print(my_product_id_0)
                elif y == 2:
                    my_product_title_2  = x
                    my_product_id_1s = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['movie_title'] == x]['movieId']
                    my_product_id_1 = 'Shirt'
                    for my_product_id_1s_next in my_product_id_1s:
                        my_product_id_1 = my_product_id_1s_next
                        break
                    print(my_product_id_1)
                elif y == 3:
                    my_product_title_3  = x
                    my_product_id_2s = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['movie_title'] == x]['movieId']
                    my_product_id_2 = 'Shirt'
                    for my_product_id_2s_next in my_product_id_2s:
                        my_product_id_2 = my_product_id_2s_next
                        break
                    print(my_product_id_2)
                elif y == 4:
                    my_product_title_4  = x
                    my_product_id_3s = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['movie_title'] == x]['movieId']
                    my_product_id_3 = 'Shirt'
                    for my_product_id_3s_next in my_product_id_3s:
                        my_product_id_3 = my_product_id_3s_next
                        break
                    print(my_product_id_3)
                elif y == 5:
                    my_product_title_5  = x
                    my_product_id_4s = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['movie_title'] == x]['movieId']
                    my_product_id_4 = 'Shirt'
                    for my_product_id_4s_next in my_product_id_4s:
                        my_product_id_4 = my_product_id_4s_next
                        break
                    print(my_product_id_4)
                elif y == 6:
                    my_product_title_6  = x
                    my_product_id_5s = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['movie_title'] == x]['movieId']
                    my_product_id_5 = 'Shirt'
                    for my_product_id_5s_next in my_product_id_5s:
                        my_product_id_5 = my_product_id_5s_next
                        break
                    print(my_product_id_5)

                elif y == 7:
                    my_product_title_7  = x
                    my_product_id_6s = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['movie_title'] == x]['movieId']
                    my_product_id_6 = 'Shirt'
                    for my_product_id_6s_next in my_product_id_6s:
                        my_product_id_6 = my_product_id_6s_next
                        break
                    print(my_product_id_6)

                elif y == 8:
                    my_product_title_8  = x
                    my_product_id_7s = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['movie_title'] == x]['movieId']
                    my_product_id_7 = 'Shirt'
                    for my_product_id_7s_next in my_product_id_7s:
                        my_product_id_7 = my_product_id_7s_next
                        break
                    print(my_product_id_7)


                elif y == 9:
                    my_product_title_9  = x
                    my_product_id_8s = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['movie_title'] == x]['movieId']
                    my_product_id_8 = 'Shirt'
                    for my_product_id_8s_next in my_product_id_8s:
                        my_product_id_8 = my_product_id_8s_next
                        break
                    print(my_product_id_8)


                elif y == 10:
                    my_product_title_10  = x
                    my_product_id_9s = ssales_parquet_file_name_df.loc[ssales_parquet_file_name_df['movie_title'] == x]['movieId']
                    my_product_id_9 = 'Shirt'
                    for my_product_id_9s_next in my_product_id_9s:
                        my_product_id_9 = my_product_id_9s_next
                        break
                    print(my_product_id_9)



                print(x)
            nudd = next_user.decode("utf-8")
            uus = nudd.replace('"', '')
            uus1 = uus[1:]
            uus2 = uus1[:-1]
            my_next_row = [x4,my_product_id_0,my_product_id_1,my_product_id_2,my_product_id_3,my_product_id_4, my_product_id_5, my_product_id_6,my_product_id_7,my_product_id_8,my_product_id_9]
            
            my_next_rowt = [x4,my_product_title_1,my_product_title_2,my_product_title_3,my_product_title_4,my_product_title_5, my_product_title_6, my_product_title_7,my_product_title_8,my_product_title_9,my_product_title_10]
            my_pred_list_lists_out_end.append(my_next_row)
            my_pred_list_lists_out_end_title.append(my_next_rowt)

    pred_file_out = '/home/ec2-user/tensor/tf_recs_2_5_v1.csv'
    with open(pred_file_out, "w") as f:
        writer = csv.writer(f)
        writer.writerows(my_pred_list_lists_out_end)
    pred_file_out = '/home/ec2-user/tensor/tf_recs_2_5_v1_TITLE.csv'
    with open(pred_file_out, "w") as f:
        writer = csv.writer(f)
        writer.writerows(my_pred_list_lists_out_end_title)


#######################################################################################################################################
# STEP 3  - RANK MODEL 
#######################################################################################################################################

def tf_recs_rank():

    import os
    import pprint
    import tempfile

    from typing import Dict, Text

    import numpy as np
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import tensorflow_recommenders as tfrs




    DATA_URL_product = '/home/ec2-user/tensor/sales_style_color_price_quint_1_19.csv'
    DATA_URL_product_df = pd.read_csv(DATA_URL_product)


    DATA_URL_product2 = '/home/ec2-user/tensor/product_style_color_price_1_19.csv'
    DATA_URL_product_df2 = pd.read_csv(DATA_URL_product2)


    ratings = tf.data.Dataset.from_tensor_slices(dict(DATA_URL_product_df)).map(lambda x: { "movie_title": x["movie_title"], "user_id": x["user_id"],"user_rating": x["rating"] })
    movies = tf.data.Dataset.from_tensor_slices(dict(DATA_URL_product_df2)).map(lambda x: x["movie_title"])


    tf.random.set_seed(42)
    shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)



    movie_titles = ratings.batch(1_000_000).map(lambda x: x["movie_title"])
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    user_ids_vocabulary.adapt(ratings.map(lambda x: x["user_id"]))


    movie_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    movie_titles_vocabulary.adapt(movies)    

    # Define user and movie models.
    user_model = tf.keras.Sequential([
        user_ids_vocabulary,
        tf.keras.layers.Embedding(user_ids_vocabulary.vocab_size(), 64)
    ])
    movie_model = tf.keras.Sequential([
        movie_titles_vocabulary,
        tf.keras.layers.Embedding(movie_titles_vocabulary.vocab_size(), 64)
    ])

    # Define your objectives.
    task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
        movies.batch(128).map(movie_model)
    )
    )


    movie_titles = ratings.batch(1_000_000).map(lambda x: x["movie_title"])
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))



    class FlagCrossMovielensModel(tfrs.models.Model):

      def __init__(self, rating_weight: float, retrieval_weight: float) -> None:
        # We take the loss weights in the constructor: this allows us to instantiate
        # several model objects with different loss weights.

        super().__init__()

        embedding_dimension = 32

        # User and movie models.
        self.movie_model: tf.keras.layers.Layer = tf.keras.Sequential([
          tf.keras.layers.StringLookup(
            vocabulary=unique_movie_titles, mask_token=None),
          tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
        ])
        self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
          tf.keras.layers.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
          tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

        self.timestamp_model: tf.keras.layers.Layer = tf.keras.Sequential([
            tf.keras.layers.Discretization(timestamp_buckets.tolist()),
            tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32)]) 
        self.timestamp_embedding = tf.keras.Sequential([
            tf.keras.layers.Discretization(timestamp_buckets.tolist()),
            tf.keras.layers.Embedding(len(timestamp_buckets) + 2, 32)])
        self.normalized_timestamp = tf.keras.layers.Normalization(
        axis=None)

        # A small model to take in user and movie embeddings and predict ratings.
        # We can make this as complicated as we want as long as we output a scalar
        # as our prediction.
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

        # The tasks.
        self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).map(self.movie_model)
            )
        )

        # The loss weights.
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

      def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the movie features and pass them into the movie model.
        movie_embeddings = self.movie_model(features["movie_title"])

        return (
            user_embeddings,
            movie_embeddings,
            # We apply the multi-layered rating model to a concatentation of
            # user and movie embeddings.
            self.rating_model(
                tf.concat([user_embeddings, movie_embeddings], axis=1)
            ),
        )

      def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

        ratings = features.pop("user_rating")

        user_embeddings, movie_embeddings, rating_predictions = self(features)

        # We compute the loss for each task.
        rating_loss = self.rating_task(
            labels=ratings,
            predictions=rating_predictions,
        )
        retrieval_loss = self.retrieval_task(user_embeddings, movie_embeddings)

        # And combine them using the loss weights.
        return (self.rating_weight * rating_loss
                + self.retrieval_weight * retrieval_loss)



    class FlagRankingModel(tf.keras.Model):

      def __init__(self):
        super().__init__()
        embedding_dimension = 32

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
          tf.keras.layers.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
          tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

        # Compute embeddings for movies.
        self.movie_embeddings = tf.keras.Sequential([
          tf.keras.layers.StringLookup(
            vocabulary=unique_movie_titles, mask_token=None),
          tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
        ])

        # Compute predictions.
        self.ratings = tf.keras.Sequential([
        # Learn multiple dense layers.
         tf.keras.layers.Dense(256, activation="relu"),
          tf.keras.layers.Dense(64, activation="relu"),
        # Make rating predictions in the final layer.
        ###########################################################
        # RATING LAYER 
        ###########################################################
          tf.keras.layers.Dense(1)
      ])

      def call(self, inputs):

        user_id, movie_title = inputs
        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.movie_embeddings(movie_title)

        return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))





    class FlagMovielensModel(tfrs.models.Model):

      def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = FlagRankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
          loss = tf.keras.losses.MeanSquaredError(),
          metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

      def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        return self.ranking_model(
            (features["user_id"], features["movie_title"]))

      def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

        ###########################################################
        # RATING LAYER 
        ###########################################################

        labels = features.pop("user_rating")

        rating_predictions = self(features)
    

        # The task computes the loss and the metrics.
        return self.task(labels=labels, predictions=rating_predictions)





    model = FlagMovielensModel()
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))



    cached_train = train.shuffle(100_000).batch(8192).cache()
    cached_test = test.batch(4096).cache()

    model.fit(cached_train, epochs=3)


    model.evaluate(cached_test, return_dict=True)

    DATA_URL_product2_tit = '/home/ec2-user/tensor/tf_recs_2_17_v1_TITLE.csv'
    DATA_URL_product2_tit2_df = pd.read_csv(DATA_URL_product2_tit)

    my_pred_list_lists_out_end_rank = []
    my_htr = ['customer_id', 'top product_title 1', 'top product_title 2', 'top product_title 3', 'top product_title 4', 'top product_title 5', 'top product_title 6']
    my_pred_list_lists_out_end_rank.append(my_htr)
    my_pred_list_lists_out_end_rank_id = []
    my_htri = ['customer_id', 'top product_id 1', 'top product_id 2', 'top product_id 3', 'top product_id 4', 'top product_id 5', 'top product_id 6']
    my_pred_list_lists_out_end_rank_id.append(my_htri)


    k = 0 
    for next_user in unique_user_ids:
        k = k +1
        
        if 0 < k and k < 10000: continue
        elif k > 20000 : break
        nus = str(next_user)
        print('nus')
        print(nus)
        #_, titles = index(np.array([nus]))
        mycid = user_ids_vocabulary.get_vocabulary()[k]
        uus = mycid.replace('"', '')
        uus1 = uus[1:]
        x = uus1[:-1]

        print('x')
        print(x)
        myxs = str(x)
        myx = int(x)
        #print(f"Top 3 recommendations for user : {titles[0, :6]}")
        my_product_id_0s = DATA_URL_product2_tit2_df.loc[DATA_URL_product2_tit2_df['customer_id'] == myx]['top product_title 1']
        print(my_product_id_0s)
        my_product_id_0 = 'Shirt'
        for my_product_id_0s_next in my_product_id_0s:
            my_product_id_0 = my_product_id_0s_next
            break
        #print(my_product_id_0)    
        my_product_id_1s = DATA_URL_product2_tit2_df.loc[DATA_URL_product2_tit2_df['customer_id'] == myx]['top product_title 2']
        my_product_id_1 = 'Shirt'
        for my_product_id_1s_next in my_product_id_1s:
            my_product_id_1 = my_product_id_1s_next
            break
        #print(my_product_id_1)    
        my_product_id_2s = DATA_URL_product2_tit2_df.loc[DATA_URL_product2_tit2_df['customer_id'] == myx]['top product_title 3']
        my_product_id_2 = 'Shirt'
        for my_product_id_2s_next in my_product_id_2s:
            my_product_id_2 = my_product_id_2s_next
            break
        #print(my_product_id_2)    
        my_product_id_3s = DATA_URL_product2_tit2_df.loc[DATA_URL_product2_tit2_df['customer_id'] == myx]['top product_title 4']
        my_product_id_3 = 'Shirt'
        for my_product_id_3s_next in my_product_id_3s:
            my_product_id_3 = my_product_id_3s_next
            break
        my_product_id_4s = DATA_URL_product2_tit2_df.loc[DATA_URL_product2_tit2_df['customer_id'] == myx]['top product_title 5']
        my_product_id_4 = 'Shirt'
        for my_product_id_4s_next in my_product_id_4s:
            my_product_id_4 = my_product_id_4s_next
            break

        my_product_id_5s = DATA_URL_product2_tit2_df.loc[DATA_URL_product2_tit2_df['customer_id'] == myx]['top product_title 6']
        my_product_id_5 = 'Shirt'
        for my_product_id_5s_next in my_product_id_5s:
            my_product_id_5 = my_product_id_5s_next
            break
        my_product_id_6s = DATA_URL_product2_tit2_df.loc[DATA_URL_product2_tit2_df['customer_id'] == myx]['top product_title 7']
        my_product_id_6 = 'Shirt'
        for my_product_id_6s_next in my_product_id_6s:
            my_product_id_6 = my_product_id_6s_next
            break
        my_product_id_7s = DATA_URL_product2_tit2_df.loc[DATA_URL_product2_tit2_df['customer_id'] == myx]['top product_title 8']
        my_product_id_7 = 'Shirt'
        for my_product_id_7s_next in my_product_id_7s:
            my_product_id_7 = my_product_id_7s_next
            break

        my_product_id_8s = DATA_URL_product2_tit2_df.loc[DATA_URL_product2_tit2_df['customer_id'] == myx]['top product_title 9']
        my_product_id_8 = 'Shirt'
        for my_product_id_8s_next in my_product_id_8s:
            my_product_id_8 = my_product_id_8s_next
            break






        test_movie_titles = [my_product_id_0,my_product_id_1,my_product_id_2,my_product_id_3,my_product_id_4,my_product_id_5,my_product_id_6,my_product_id_7,my_product_id_8]
        test_ratings = {}
        for movie_title in test_movie_titles:
            test_ratings[movie_title] = model({ "user_id": np.array([user_ids_vocabulary.get_vocabulary()[k]]), "movie_title": np.array([movie_title]) })

        print("My Ratings:")
        r =0 
        for title, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
            print(f"{title}: {score}")
            r = r + 1
            if r == 1 : 
                fp1 = title
                my_product_id_2s = DATA_URL_product_df2.loc[DATA_URL_product_df2['movie_title'] == fp1]['movieId']
                my_product_id_1_2 = 'Shirt'
                for my_product_id_2s_next in my_product_id_2s:
                    my_product_id_1_2 = my_product_id_2s_next
                    break
                #print("my_product_id_2")
                #print(my_product_id_1_2)





            elif r ==2 :
                fp2 = title
                my_product_id_22s = DATA_URL_product_df2.loc[DATA_URL_product_df2['movie_title'] == fp2]['movieId']
                my_product_id_2_2 = 'Shirt'
                for my_product_id_2s_next in my_product_id_22s:
                    my_product_id_2_2 = my_product_id_2s_next
                    break
                #print("my_product_id_2")
                #print(my_product_id_2_2)

            elif r ==3 :
                fp3 = title
                my_product_id_32s = DATA_URL_product_df2.loc[DATA_URL_product_df2['movie_title'] == fp3]['movieId']
                my_product_id_3_2 = 'Shirt'
                for my_product_id_2s_next in my_product_id_32s:
                    my_product_id_3_2 = my_product_id_2s_next
                    break
                #print("my_product_id_2")
                #print(my_product_id_3_2)

            elif r ==4 :
                fp4 = title
                my_product_id_42s = DATA_URL_product_df2.loc[DATA_URL_product_df2['movie_title'] == fp4]['movieId']
                my_product_id_4_2 = 'Shirt'
                for my_product_id_2s_next in my_product_id_42s:
                    my_product_id_4_2 = my_product_id_2s_next
                    break
                #print("my_product_id_2")
                #print(my_product_id_4_2)

            elif r ==5 :
                fp5 = title
                my_product_id_52s = DATA_URL_product_df2.loc[DATA_URL_product_df2['movie_title'] == fp5]['movieId']
                my_product_id_5_2 = 'Shirt'
                for my_product_id_2s_next in my_product_id_52s:
                    my_product_id_5_2 = my_product_id_2s_next
                    break
                #print("my_product_id_2")
                #print(my_product_id_5_2)

            elif r ==6 :
                fp6 = title
                my_product_id_62s = DATA_URL_product_df2.loc[DATA_URL_product_df2['movie_title'] == fp6]['movieId']
                my_product_id_6_2 = 'Shirt'
                for my_product_id_2s_next in my_product_id_62s:
                    my_product_id_6_2 = my_product_id_2s_next
                    break


        mynrf = [x,fp1,fp2,fp3,fp4,fp5,fp6]
        my_pred_list_lists_out_end_rank.append(mynrf)

        mynrfi = [x,my_product_id_1_2,my_product_id_2_2,my_product_id_3_2,my_product_id_4_2,my_product_id_5_2,my_product_id_6_2]
        my_pred_list_lists_out_end_rank_id.append(mynrfi)



    pred_file_out = '/home/ec2-user/tensor/tf_recs_2_17_FINAL_20000.csv'
    with open(pred_file_out, "w") as f:
        writer = csv.writer(f)
        writer.writerows(my_pred_list_lists_out_end_rank)

    pred_file_out = '/home/ec2-user/tensor/tf_recs_2_17_FINAL_ID_20000.csv'
    with open(pred_file_out, "w") as f:
        writer = csv.writer(f)
        writer.writerows(my_pred_list_lists_out_end_rank_id)

 


def tf_recs_rank_deep():

    import os
    import pprint
    import tempfile

    from typing import Dict, Text

    import numpy as np
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import tensorflow_recommenders as tfrs




    DATA_URL_product = '/home/ec2-user/tensor/sales_style_color_price_quint_1_19.csv'
    DATA_URL_product_df = pd.read_csv(DATA_URL_product)


    DATA_URL_product2 = '/home/ec2-user/tensor/product_style_color_price_1_19.csv'
    DATA_URL_product_df2 = pd.read_csv(DATA_URL_product2)


    ratings = tf.data.Dataset.from_tensor_slices(dict(DATA_URL_product_df)).map(lambda x: { "movie_title": x["movie_title"], "user_id": x["user_id"],"user_rating": x["rating"] })
    movies = tf.data.Dataset.from_tensor_slices(dict(DATA_URL_product_df2)).map(lambda x: x["movie_title"])


    tf.random.set_seed(42)
    shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)



    movie_titles = ratings.batch(1_000_000).map(lambda x: x["movie_title"])
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    user_ids_vocabulary.adapt(ratings.map(lambda x: x["user_id"]))


    movie_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    movie_titles_vocabulary.adapt(movies)    

    # Define user and movie models.
    user_model = tf.keras.Sequential([
        user_ids_vocabulary,
        tf.keras.layers.Embedding(user_ids_vocabulary.vocab_size(), 64)
    ])
    movie_model = tf.keras.Sequential([
        movie_titles_vocabulary,
        tf.keras.layers.Embedding(movie_titles_vocabulary.vocab_size(), 64)
    ])

    # Define your objectives.
    task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
        movies.batch(128).map(movie_model)
    )
    )


    movie_titles = ratings.batch(1_000_000).map(lambda x: x["movie_title"])
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))



    class FlagCrossMovielensModel(tfrs.models.Model):

      def __init__(self, rating_weight: float, retrieval_weight: float) -> None:
        # We take the loss weights in the constructor: this allows us to instantiate
        # several model objects with different loss weights.

        super().__init__()

        embedding_dimension = 32

        # User and movie models.
        self.movie_model: tf.keras.layers.Layer = tf.keras.Sequential([
          tf.keras.layers.StringLookup(
            vocabulary=unique_movie_titles, mask_token=None),
          tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
        ])
        self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
          tf.keras.layers.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
          tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

        self.timestamp_model: tf.keras.layers.Layer = tf.keras.Sequential([
            tf.keras.layers.Discretization(timestamp_buckets.tolist()),
            tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32)]) 
        self.timestamp_embedding = tf.keras.Sequential([
            tf.keras.layers.Discretization(timestamp_buckets.tolist()),
            tf.keras.layers.Embedding(len(timestamp_buckets) + 2, 32)])
        self.normalized_timestamp = tf.keras.layers.Normalization(
        axis=None)

        # A small model to take in user and movie embeddings and predict ratings.
        # We can make this as complicated as we want as long as we output a scalar
        # as our prediction.
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

        # The tasks.
        self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).map(self.movie_model)
            )
        )

        # The loss weights.
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight

      def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the movie features and pass them into the movie model.
        movie_embeddings = self.movie_model(features["movie_title"])

        return (
            user_embeddings,
            movie_embeddings,
            # We apply the multi-layered rating model to a concatentation of
            # user and movie embeddings.
            self.rating_model(
                tf.concat([user_embeddings, movie_embeddings], axis=1)
            ),
        )

      def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

        ratings = features.pop("user_rating")

        user_embeddings, movie_embeddings, rating_predictions = self(features)

        # We compute the loss for each task.
        rating_loss = self.rating_task(
            labels=ratings,
            predictions=rating_predictions,
        )
        retrieval_loss = self.retrieval_task(user_embeddings, movie_embeddings)

        # And combine them using the loss weights.
        return (self.rating_weight * rating_loss
                + self.retrieval_weight * retrieval_loss)



    class FlagRankingModel(tf.keras.Model):

      def __init__(self):
        super().__init__()
        embedding_dimension = 32

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
          tf.keras.layers.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
          tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

        # Compute embeddings for movies.
        self.movie_embeddings = tf.keras.Sequential([
          tf.keras.layers.StringLookup(
            vocabulary=unique_movie_titles, mask_token=None),
          tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
        ])

        # Compute predictions.
        self.ratings = tf.keras.Sequential([
        # Learn multiple dense layers.
         tf.keras.layers.Dense(256, activation="relu"),
          tf.keras.layers.Dense(64, activation="relu"),
        # Make rating predictions in the final layer.
        ###########################################################
        # RATING LAYER 
        ###########################################################
          tf.keras.layers.Dense(1)
      ])

      def call(self, inputs):

        user_id, movie_title = inputs
        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.movie_embeddings(movie_title)

        return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))





    class FlagMovielensModel(tfrs.models.Model):

      def __init__(self):
        super().__init__()
        self.ranking_model: tf.keras.Model = FlagRankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
          loss = tf.keras.losses.MeanSquaredError(),
          metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

      def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        return self.ranking_model(
            (features["user_id"], features["movie_title"]))

      def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

        ###########################################################
        # RATING LAYER 
        ###########################################################

        labels = features.pop("user_rating")

        rating_predictions = self(features)
    

        # The task computes the loss and the metrics.
        return self.task(labels=labels, predictions=rating_predictions)





    model = FlagMovielensModel()
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))



    cached_train = train.shuffle(100_000).batch(8192).cache()
    cached_test = test.batch(4096).cache()

    model.fit(cached_train, epochs=3)


    model.evaluate(cached_test, return_dict=True)

    DATA_URL_product2_tit = '/home/ec2-user/tensor/tf_recs_2_17_v1_TITLE.csv'
    DATA_URL_product2_tit2_df = pd.read_csv(DATA_URL_product2_tit)

    my_pred_list_lists_out_end_rank = []
    my_htr = ['customer_id', 'top product_title 1', 'top product_title 2', 'top product_title 3', 'top product_title 4', 'top product_title 5', 'top product_title 6']
    my_pred_list_lists_out_end_rank.append(my_htr)
    my_pred_list_lists_out_end_rank_id = []
    my_htri = ['customer_id', 'top product_id 1', 'top product_id 2', 'top product_id 3', 'top product_id 4', 'top product_id 5', 'top product_id 6']
    my_pred_list_lists_out_end_rank_id.append(my_htri)


    k = 0 
    for next_user in unique_user_ids:
        k = k +1
        if k > 10000 : break    
        #if 0 < k and k < 10000: continue
        #elif k > 20000 : break
        nus = str(next_user)
        print('nus')
        print(nus)
        #_, titles = index(np.array([nus]))
        mycid = user_ids_vocabulary.get_vocabulary()[k]
        uus = mycid.replace('"', '')
        uus1 = uus[1:]
        x = uus1[:-1]

        print('x')
        print(x)
        myxs = str(x)
        myx = int(x)
        #print(f"Top 3 recommendations for user : {titles[0, :6]}")
        my_product_id_0s = DATA_URL_product2_tit2_df.loc[DATA_URL_product2_tit2_df['customer_id'] == myx]['top product_title 1']
        print(my_product_id_0s)
        my_product_id_0 = 'Shirt'
        for my_product_id_0s_next in my_product_id_0s:
            my_product_id_0 = my_product_id_0s_next
            break
        #print(my_product_id_0)    
        my_product_id_1s = DATA_URL_product2_tit2_df.loc[DATA_URL_product2_tit2_df['customer_id'] == myx]['top product_title 2']
        my_product_id_1 = 'Shirt'
        for my_product_id_1s_next in my_product_id_1s:
            my_product_id_1 = my_product_id_1s_next
            break
        #print(my_product_id_1)    
        my_product_id_2s = DATA_URL_product2_tit2_df.loc[DATA_URL_product2_tit2_df['customer_id'] == myx]['top product_title 3']
        my_product_id_2 = 'Shirt'
        for my_product_id_2s_next in my_product_id_2s:
            my_product_id_2 = my_product_id_2s_next
            break
        #print(my_product_id_2)    
        my_product_id_3s = DATA_URL_product2_tit2_df.loc[DATA_URL_product2_tit2_df['customer_id'] == myx]['top product_title 4']
        my_product_id_3 = 'Shirt'
        for my_product_id_3s_next in my_product_id_3s:
            my_product_id_3 = my_product_id_3s_next
            break
        my_product_id_4s = DATA_URL_product2_tit2_df.loc[DATA_URL_product2_tit2_df['customer_id'] == myx]['top product_title 5']
        my_product_id_4 = 'Shirt'
        for my_product_id_4s_next in my_product_id_4s:
            my_product_id_4 = my_product_id_4s_next
            break

        my_product_id_5s = DATA_URL_product2_tit2_df.loc[DATA_URL_product2_tit2_df['customer_id'] == myx]['top product_title 6']
        my_product_id_5 = 'Shirt'
        for my_product_id_5s_next in my_product_id_5s:
            my_product_id_5 = my_product_id_5s_next
            break
        my_product_id_6s = DATA_URL_product2_tit2_df.loc[DATA_URL_product2_tit2_df['customer_id'] == myx]['top product_title 7']
        my_product_id_6 = 'Shirt'
        for my_product_id_6s_next in my_product_id_6s:
            my_product_id_6 = my_product_id_6s_next
            break
        my_product_id_7s = DATA_URL_product2_tit2_df.loc[DATA_URL_product2_tit2_df['customer_id'] == myx]['top product_title 8']
        my_product_id_7 = 'Shirt'
        for my_product_id_7s_next in my_product_id_7s:
            my_product_id_7 = my_product_id_7s_next
            break

        my_product_id_8s = DATA_URL_product2_tit2_df.loc[DATA_URL_product2_tit2_df['customer_id'] == myx]['top product_title 9']
        my_product_id_8 = 'Shirt'
        for my_product_id_8s_next in my_product_id_8s:
            my_product_id_8 = my_product_id_8s_next
            break






        test_movie_titles = [my_product_id_0,my_product_id_1,my_product_id_2,my_product_id_3,my_product_id_4,my_product_id_5,my_product_id_6,my_product_id_7,my_product_id_8]
        test_ratings = {}
        for movie_title in test_movie_titles:
            test_ratings[movie_title] = model({ "user_id": np.array([user_ids_vocabulary.get_vocabulary()[k]]), "movie_title": np.array([movie_title]) })

        print("My Ratings:")
        r =0 
        for title, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
            print(f"{title}: {score}")
            r = r + 1
            if r == 1 : 
                fp1 = title
                my_product_id_2s = DATA_URL_product_df2.loc[DATA_URL_product_df2['movie_title'] == fp1]['movieId']
                my_product_id_1_2 = 'Shirt'
                for my_product_id_2s_next in my_product_id_2s:
                    my_product_id_1_2 = my_product_id_2s_next
                    break
                #print("my_product_id_2")
                #print(my_product_id_1_2)





            elif r ==2 :
                fp2 = title
                my_product_id_22s = DATA_URL_product_df2.loc[DATA_URL_product_df2['movie_title'] == fp2]['movieId']
                my_product_id_2_2 = 'Shirt'
                for my_product_id_2s_next in my_product_id_22s:
                    my_product_id_2_2 = my_product_id_2s_next
                    break
                #print("my_product_id_2")
                #print(my_product_id_2_2)

            elif r ==3 :
                fp3 = title
                my_product_id_32s = DATA_URL_product_df2.loc[DATA_URL_product_df2['movie_title'] == fp3]['movieId']
                my_product_id_3_2 = 'Shirt'
                for my_product_id_2s_next in my_product_id_32s:
                    my_product_id_3_2 = my_product_id_2s_next
                    break
                #print("my_product_id_2")
                #print(my_product_id_3_2)

            elif r ==4 :
                fp4 = title
                my_product_id_42s = DATA_URL_product_df2.loc[DATA_URL_product_df2['movie_title'] == fp4]['movieId']
                my_product_id_4_2 = 'Shirt'
                for my_product_id_2s_next in my_product_id_42s:
                    my_product_id_4_2 = my_product_id_2s_next
                    break
                #print("my_product_id_2")
                #print(my_product_id_4_2)

            elif r ==5 :
                fp5 = title
                my_product_id_52s = DATA_URL_product_df2.loc[DATA_URL_product_df2['movie_title'] == fp5]['movieId']
                my_product_id_5_2 = 'Shirt'
                for my_product_id_2s_next in my_product_id_52s:
                    my_product_id_5_2 = my_product_id_2s_next
                    break
                #print("my_product_id_2")
                #print(my_product_id_5_2)

            elif r ==6 :
                fp6 = title
                my_product_id_62s = DATA_URL_product_df2.loc[DATA_URL_product_df2['movie_title'] == fp6]['movieId']
                my_product_id_6_2 = 'Shirt'
                for my_product_id_2s_next in my_product_id_62s:
                    my_product_id_6_2 = my_product_id_2s_next
                    break


        mynrf = [x,fp1,fp2,fp3,fp4,fp5,fp6]
        my_pred_list_lists_out_end_rank.append(mynrf)

        mynrfi = [x,my_product_id_1_2,my_product_id_2_2,my_product_id_3_2,my_product_id_4_2,my_product_id_5_2,my_product_id_6_2]
        my_pred_list_lists_out_end_rank_id.append(mynrfi)



    model = FlagCrossMovielensModel(rating_weight=1.0, retrieval_weight=0.0)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

    cached_train = train.shuffle(100_000).batch(8192).cache()
    cached_test = test.batch(4096).cache()


    model.fit(cached_train, epochs=3)
    metrics = model.evaluate(cached_test, return_dict=True)

    print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
    print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")



    model = FlagCrossMovielensModel(rating_weight=0.0, retrieval_weight=1.0)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

    model.fit(cached_train, epochs=3)
    metrics = model.evaluate(cached_test, return_dict=True)

    print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
    print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")


    model = FlagCrossMovielensModel(rating_weight=1.0, retrieval_weight=1.0)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

    model.fit(cached_train, epochs=3)
    metrics = model.evaluate(cached_test, return_dict=True)

    print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
    print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")


  
    pred_file_out = '/home/ec2-user/tensor/tf_recs_2_17_FINAL_10000.csv'
    with open(pred_file_out, "w") as f:
        writer = csv.writer(f)
        writer.writerows(my_pred_list_lists_out_end_rank)

    pred_file_out = '/home/ec2-user/tensor/tf_recs_2_17_FINAL_ID_10000.csv'
    with open(pred_file_out, "w") as f:
        writer = csv.writer(f)
        writer.writerows(my_pred_list_lists_out_end_rank_id)

 


def main():

    tf_data_prep()
    tf_recs_ret_deep()
    tf_recs_rank()


if __name__ == '__main__':
    main()





