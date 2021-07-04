{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Fake news Detection\n",
    "### Final year project"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Importing required library\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "# ml lib \n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.metrics import classification_report\n",
    "import re\n",
    "import string"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Inserting fake and real dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_fake = pd.read_csv('cvs/Fake.csv')\n",
    "df_true = pd.read_csv('cvs/True.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>subject</th>\n",
       "      <th>date</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Donald Trump Sends Out Embarrassing New Year’...</td>\n",
       "      <td>Donald Trump just couldn t wish all Americans ...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 31, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Drunk Bragging Trump Staffer Started Russian ...</td>\n",
       "      <td>House Intelligence Committee Chairman Devin Nu...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 31, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Sheriff David Clarke Becomes An Internet Joke...</td>\n",
       "      <td>On Friday, it was revealed that former Milwauk...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 30, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Trump Is So Obsessed He Even Has Obama’s Name...</td>\n",
       "      <td>On Christmas day, Donald Trump announced that ...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 29, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Pope Francis Just Called Out Donald Trump Dur...</td>\n",
       "      <td>Pope Francis used his annual Christmas Day mes...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 25, 2017</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               title  \\\n",
       "0   Donald Trump Sends Out Embarrassing New Year’...   \n",
       "1   Drunk Bragging Trump Staffer Started Russian ...   \n",
       "2   Sheriff David Clarke Becomes An Internet Joke...   \n",
       "3   Trump Is So Obsessed He Even Has Obama’s Name...   \n",
       "4   Pope Francis Just Called Out Donald Trump Dur...   \n",
       "\n",
       "                                                text subject  \\\n",
       "0  Donald Trump just couldn t wish all Americans ...    News   \n",
       "1  House Intelligence Committee Chairman Devin Nu...    News   \n",
       "2  On Friday, it was revealed that former Milwauk...    News   \n",
       "3  On Christmas day, Donald Trump announced that ...    News   \n",
       "4  Pope Francis used his annual Christmas Day mes...    News   \n",
       "\n",
       "                date  \n",
       "0  December 31, 2017  \n",
       "1  December 31, 2017  \n",
       "2  December 30, 2017  \n",
       "3  December 29, 2017  \n",
       "4  December 25, 2017  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_fake.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>subject</th>\n",
       "      <th>date</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>As U.S. budget fight looms, Republicans flip t...</td>\n",
       "      <td>WASHINGTON (Reuters) - The head of a conservat...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 31, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>U.S. military to accept transgender recruits o...</td>\n",
       "      <td>WASHINGTON (Reuters) - Transgender people will...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 29, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Senior U.S. Republican senator: 'Let Mr. Muell...</td>\n",
       "      <td>WASHINGTON (Reuters) - The special counsel inv...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 31, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>FBI Russia probe helped by Australian diplomat...</td>\n",
       "      <td>WASHINGTON (Reuters) - Trump campaign adviser ...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 30, 2017</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Trump wants Postal Service to charge 'much mor...</td>\n",
       "      <td>SEATTLE/WASHINGTON (Reuters) - President Donal...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 29, 2017</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               title  \\\n",
       "0  As U.S. budget fight looms, Republicans flip t...   \n",
       "1  U.S. military to accept transgender recruits o...   \n",
       "2  Senior U.S. Republican senator: 'Let Mr. Muell...   \n",
       "3  FBI Russia probe helped by Australian diplomat...   \n",
       "4  Trump wants Postal Service to charge 'much mor...   \n",
       "\n",
       "                                                text       subject  \\\n",
       "0  WASHINGTON (Reuters) - The head of a conservat...  politicsNews   \n",
       "1  WASHINGTON (Reuters) - Transgender people will...  politicsNews   \n",
       "2  WASHINGTON (Reuters) - The special counsel inv...  politicsNews   \n",
       "3  WASHINGTON (Reuters) - Trump campaign adviser ...  politicsNews   \n",
       "4  SEATTLE/WASHINGTON (Reuters) - President Donal...  politicsNews   \n",
       "\n",
       "                 date  \n",
       "0  December 31, 2017   \n",
       "1  December 29, 2017   \n",
       "2  December 31, 2017   \n",
       "3  December 30, 2017   \n",
       "4  December 29, 2017   "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_true.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Inserting a column called \"class\" for fake and real news dataset to categories fake and true news. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>subject</th>\n",
       "      <th>date</th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>As U.S. budget fight looms, Republicans flip t...</td>\n",
       "      <td>WASHINGTON (Reuters) - The head of a conservat...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 31, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>U.S. military to accept transgender recruits o...</td>\n",
       "      <td>WASHINGTON (Reuters) - Transgender people will...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 29, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Senior U.S. Republican senator: 'Let Mr. Muell...</td>\n",
       "      <td>WASHINGTON (Reuters) - The special counsel inv...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 31, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>FBI Russia probe helped by Australian diplomat...</td>\n",
       "      <td>WASHINGTON (Reuters) - Trump campaign adviser ...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 30, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Trump wants Postal Service to charge 'much mor...</td>\n",
       "      <td>SEATTLE/WASHINGTON (Reuters) - President Donal...</td>\n",
       "      <td>politicsNews</td>\n",
       "      <td>December 29, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               title  \\\n",
       "0  As U.S. budget fight looms, Republicans flip t...   \n",
       "1  U.S. military to accept transgender recruits o...   \n",
       "2  Senior U.S. Republican senator: 'Let Mr. Muell...   \n",
       "3  FBI Russia probe helped by Australian diplomat...   \n",
       "4  Trump wants Postal Service to charge 'much mor...   \n",
       "\n",
       "                                                text       subject  \\\n",
       "0  WASHINGTON (Reuters) - The head of a conservat...  politicsNews   \n",
       "1  WASHINGTON (Reuters) - Transgender people will...  politicsNews   \n",
       "2  WASHINGTON (Reuters) - The special counsel inv...  politicsNews   \n",
       "3  WASHINGTON (Reuters) - Trump campaign adviser ...  politicsNews   \n",
       "4  SEATTLE/WASHINGTON (Reuters) - President Donal...  politicsNews   \n",
       "\n",
       "                 date  class  \n",
       "0  December 31, 2017       1  \n",
       "1  December 29, 2017       1  \n",
       "2  December 31, 2017       1  \n",
       "3  December 30, 2017       1  \n",
       "4  December 29, 2017       1  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_fake[\"class\"] = 0\n",
    "df_true[\"class\"] = 1\n",
    "df_true.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Removing last 10 rows from both the dataset, for manual testing  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((23481, 5), (21417, 5))"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_fake.shape, df_true.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_fake_manual_testing = df_fake.tail(10)\n",
    "# for i in range(23480,23470,-1):\n",
    "#     df_fake.drop([i], axis = 0, inplace = True)\n",
    "df_true_manual_testing = df_true.tail(10)\n",
    "# for i in range(21416,21406,-1):\n",
    "#     df_true.drop([i], axis = 0, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((23481, 5), (21417, 5))"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_fake.shape, df_true.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Merging the manual testing dataframe in single dataset and save it in a csv file"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-9-3aaf8ec2aad1>:1: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df_fake_manual_testing[\"class\"] = 0\n",
      "<ipython-input-9-3aaf8ec2aad1>:2: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  df_true_manual_testing[\"class\"] = 1\n"
     ]
    }
   ],
   "source": [
    "df_fake_manual_testing[\"class\"] = 0\n",
    "df_true_manual_testing[\"class\"] = 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>subject</th>\n",
       "      <th>date</th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>23471</th>\n",
       "      <td>Seven Iranians freed in the prisoner swap have...</td>\n",
       "      <td>21st Century Wire says This week, the historic...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 20, 2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23472</th>\n",
       "      <td>#Hashtag Hell &amp; The Fake Left</td>\n",
       "      <td>By Dady Chery and Gilbert MercierAll writers ...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 19, 2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23473</th>\n",
       "      <td>Astroturfing: Journalist Reveals Brainwashing ...</td>\n",
       "      <td>Vic Bishop Waking TimesOur reality is carefull...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 19, 2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23474</th>\n",
       "      <td>The New American Century: An Era of Fraud</td>\n",
       "      <td>Paul Craig RobertsIn the last years of the 20t...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 19, 2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23475</th>\n",
       "      <td>Hillary Clinton: ‘Israel First’ (and no peace ...</td>\n",
       "      <td>Robert Fantina CounterpunchAlthough the United...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 18, 2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23476</th>\n",
       "      <td>McPain: John McCain Furious That Iran Treated ...</td>\n",
       "      <td>21st Century Wire says As 21WIRE reported earl...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 16, 2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23477</th>\n",
       "      <td>JUSTICE? Yahoo Settles E-mail Privacy Class-ac...</td>\n",
       "      <td>21st Century Wire says It s a familiar theme. ...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 16, 2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23478</th>\n",
       "      <td>Sunnistan: US and Allied ‘Safe Zone’ Plan to T...</td>\n",
       "      <td>Patrick Henningsen  21st Century WireRemember ...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 15, 2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23479</th>\n",
       "      <td>How to Blow $700 Million: Al Jazeera America F...</td>\n",
       "      <td>21st Century Wire says Al Jazeera America will...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 14, 2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23480</th>\n",
       "      <td>10 U.S. Navy Sailors Held by Iranian Military ...</td>\n",
       "      <td>21st Century Wire says As 21WIRE predicted in ...</td>\n",
       "      <td>Middle-east</td>\n",
       "      <td>January 12, 2016</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                   title  \\\n",
       "23471  Seven Iranians freed in the prisoner swap have...   \n",
       "23472                      #Hashtag Hell & The Fake Left   \n",
       "23473  Astroturfing: Journalist Reveals Brainwashing ...   \n",
       "23474          The New American Century: An Era of Fraud   \n",
       "23475  Hillary Clinton: ‘Israel First’ (and no peace ...   \n",
       "23476  McPain: John McCain Furious That Iran Treated ...   \n",
       "23477  JUSTICE? Yahoo Settles E-mail Privacy Class-ac...   \n",
       "23478  Sunnistan: US and Allied ‘Safe Zone’ Plan to T...   \n",
       "23479  How to Blow $700 Million: Al Jazeera America F...   \n",
       "23480  10 U.S. Navy Sailors Held by Iranian Military ...   \n",
       "\n",
       "                                                    text      subject  \\\n",
       "23471  21st Century Wire says This week, the historic...  Middle-east   \n",
       "23472   By Dady Chery and Gilbert MercierAll writers ...  Middle-east   \n",
       "23473  Vic Bishop Waking TimesOur reality is carefull...  Middle-east   \n",
       "23474  Paul Craig RobertsIn the last years of the 20t...  Middle-east   \n",
       "23475  Robert Fantina CounterpunchAlthough the United...  Middle-east   \n",
       "23476  21st Century Wire says As 21WIRE reported earl...  Middle-east   \n",
       "23477  21st Century Wire says It s a familiar theme. ...  Middle-east   \n",
       "23478  Patrick Henningsen  21st Century WireRemember ...  Middle-east   \n",
       "23479  21st Century Wire says Al Jazeera America will...  Middle-east   \n",
       "23480  21st Century Wire says As 21WIRE predicted in ...  Middle-east   \n",
       "\n",
       "                   date  class  \n",
       "23471  January 20, 2016      0  \n",
       "23472  January 19, 2016      0  \n",
       "23473  January 19, 2016      0  \n",
       "23474  January 19, 2016      0  \n",
       "23475  January 18, 2016      0  \n",
       "23476  January 16, 2016      0  \n",
       "23477  January 16, 2016      0  \n",
       "23478  January 15, 2016      0  \n",
       "23479  January 14, 2016      0  \n",
       "23480  January 12, 2016      0  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_fake_manual_testing.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>subject</th>\n",
       "      <th>date</th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>21407</th>\n",
       "      <td>Mata Pires, owner of embattled Brazil builder ...</td>\n",
       "      <td>SAO PAULO (Reuters) - Cesar Mata Pires, the ow...</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>August 22, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21408</th>\n",
       "      <td>U.S., North Korea clash at U.N. forum over nuc...</td>\n",
       "      <td>GENEVA (Reuters) - North Korea and the United ...</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>August 22, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21409</th>\n",
       "      <td>U.S., North Korea clash at U.N. arms forum on ...</td>\n",
       "      <td>GENEVA (Reuters) - North Korea and the United ...</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>August 22, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21410</th>\n",
       "      <td>Headless torso could belong to submarine journ...</td>\n",
       "      <td>COPENHAGEN (Reuters) - Danish police said on T...</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>August 22, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21411</th>\n",
       "      <td>North Korea shipments to Syria chemical arms a...</td>\n",
       "      <td>UNITED NATIONS (Reuters) - Two North Korean sh...</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>August 21, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21412</th>\n",
       "      <td>'Fully committed' NATO backs new U.S. approach...</td>\n",
       "      <td>BRUSSELS (Reuters) - NATO allies on Tuesday we...</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>August 22, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21413</th>\n",
       "      <td>LexisNexis withdrew two products from Chinese ...</td>\n",
       "      <td>LONDON (Reuters) - LexisNexis, a provider of l...</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>August 22, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21414</th>\n",
       "      <td>Minsk cultural hub becomes haven from authorities</td>\n",
       "      <td>MINSK (Reuters) - In the shadow of disused Sov...</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>August 22, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21415</th>\n",
       "      <td>Vatican upbeat on possibility of Pope Francis ...</td>\n",
       "      <td>MOSCOW (Reuters) - Vatican Secretary of State ...</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>August 22, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21416</th>\n",
       "      <td>Indonesia to buy $1.14 billion worth of Russia...</td>\n",
       "      <td>JAKARTA (Reuters) - Indonesia will buy 11 Sukh...</td>\n",
       "      <td>worldnews</td>\n",
       "      <td>August 22, 2017</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                   title  \\\n",
       "21407  Mata Pires, owner of embattled Brazil builder ...   \n",
       "21408  U.S., North Korea clash at U.N. forum over nuc...   \n",
       "21409  U.S., North Korea clash at U.N. arms forum on ...   \n",
       "21410  Headless torso could belong to submarine journ...   \n",
       "21411  North Korea shipments to Syria chemical arms a...   \n",
       "21412  'Fully committed' NATO backs new U.S. approach...   \n",
       "21413  LexisNexis withdrew two products from Chinese ...   \n",
       "21414  Minsk cultural hub becomes haven from authorities   \n",
       "21415  Vatican upbeat on possibility of Pope Francis ...   \n",
       "21416  Indonesia to buy $1.14 billion worth of Russia...   \n",
       "\n",
       "                                                    text    subject  \\\n",
       "21407  SAO PAULO (Reuters) - Cesar Mata Pires, the ow...  worldnews   \n",
       "21408  GENEVA (Reuters) - North Korea and the United ...  worldnews   \n",
       "21409  GENEVA (Reuters) - North Korea and the United ...  worldnews   \n",
       "21410  COPENHAGEN (Reuters) - Danish police said on T...  worldnews   \n",
       "21411  UNITED NATIONS (Reuters) - Two North Korean sh...  worldnews   \n",
       "21412  BRUSSELS (Reuters) - NATO allies on Tuesday we...  worldnews   \n",
       "21413  LONDON (Reuters) - LexisNexis, a provider of l...  worldnews   \n",
       "21414  MINSK (Reuters) - In the shadow of disused Sov...  worldnews   \n",
       "21415  MOSCOW (Reuters) - Vatican Secretary of State ...  worldnews   \n",
       "21416  JAKARTA (Reuters) - Indonesia will buy 11 Sukh...  worldnews   \n",
       "\n",
       "                   date  class  \n",
       "21407  August 22, 2017       1  \n",
       "21408  August 22, 2017       1  \n",
       "21409  August 22, 2017       1  \n",
       "21410  August 22, 2017       1  \n",
       "21411  August 21, 2017       1  \n",
       "21412  August 22, 2017       1  \n",
       "21413  August 22, 2017       1  \n",
       "21414  August 22, 2017       1  \n",
       "21415  August 22, 2017       1  \n",
       "21416  August 22, 2017       1  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_true_manual_testing.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## creating a manual file"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)\n",
    "df_manual_testing.to_csv(\"testing.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Merging the main fake and true dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>text</th>\n",
       "      <th>subject</th>\n",
       "      <th>date</th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Donald Trump Sends Out Embarrassing New Year’...</td>\n",
       "      <td>Donald Trump just couldn t wish all Americans ...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 31, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Drunk Bragging Trump Staffer Started Russian ...</td>\n",
       "      <td>House Intelligence Committee Chairman Devin Nu...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 31, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Sheriff David Clarke Becomes An Internet Joke...</td>\n",
       "      <td>On Friday, it was revealed that former Milwauk...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 30, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Trump Is So Obsessed He Even Has Obama’s Name...</td>\n",
       "      <td>On Christmas day, Donald Trump announced that ...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 29, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Pope Francis Just Called Out Donald Trump Dur...</td>\n",
       "      <td>Pope Francis used his annual Christmas Day mes...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 25, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Racist Alabama Cops Brutalize Black Boy While...</td>\n",
       "      <td>The number of cases of cops brutalizing and ki...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 25, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Fresh Off The Golf Course, Trump Lashes Out A...</td>\n",
       "      <td>Donald Trump spent a good portion of his day a...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 23, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>Trump Said Some INSANELY Racist Stuff Inside ...</td>\n",
       "      <td>In the wake of yet another court decision that...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 23, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>Former CIA Director Slams Trump Over UN Bully...</td>\n",
       "      <td>Many people have raised the alarm regarding th...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 22, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>WATCH: Brand-New Pro-Trump Ad Features So Muc...</td>\n",
       "      <td>Just when you might have thought we d get a br...</td>\n",
       "      <td>News</td>\n",
       "      <td>December 21, 2017</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               title  \\\n",
       "0   Donald Trump Sends Out Embarrassing New Year’...   \n",
       "1   Drunk Bragging Trump Staffer Started Russian ...   \n",
       "2   Sheriff David Clarke Becomes An Internet Joke...   \n",
       "3   Trump Is So Obsessed He Even Has Obama’s Name...   \n",
       "4   Pope Francis Just Called Out Donald Trump Dur...   \n",
       "5   Racist Alabama Cops Brutalize Black Boy While...   \n",
       "6   Fresh Off The Golf Course, Trump Lashes Out A...   \n",
       "7   Trump Said Some INSANELY Racist Stuff Inside ...   \n",
       "8   Former CIA Director Slams Trump Over UN Bully...   \n",
       "9   WATCH: Brand-New Pro-Trump Ad Features So Muc...   \n",
       "\n",
       "                                                text subject  \\\n",
       "0  Donald Trump just couldn t wish all Americans ...    News   \n",
       "1  House Intelligence Committee Chairman Devin Nu...    News   \n",
       "2  On Friday, it was revealed that former Milwauk...    News   \n",
       "3  On Christmas day, Donald Trump announced that ...    News   \n",
       "4  Pope Francis used his annual Christmas Day mes...    News   \n",
       "5  The number of cases of cops brutalizing and ki...    News   \n",
       "6  Donald Trump spent a good portion of his day a...    News   \n",
       "7  In the wake of yet another court decision that...    News   \n",
       "8  Many people have raised the alarm regarding th...    News   \n",
       "9  Just when you might have thought we d get a br...    News   \n",
       "\n",
       "                date  class  \n",
       "0  December 31, 2017      0  \n",
       "1  December 31, 2017      0  \n",
       "2  December 30, 2017      0  \n",
       "3  December 29, 2017      0  \n",
       "4  December 25, 2017      0  \n",
       "5  December 25, 2017      0  \n",
       "6  December 23, 2017      0  \n",
       "7  December 23, 2017      0  \n",
       "8  December 22, 2017      0  \n",
       "9  December 21, 2017      0  "
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_marge = pd.concat([df_fake, df_true], axis =0 )\n",
    "\n",
    "df_marge.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['title', 'text', 'subject', 'date', 'class'], dtype='object')"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_marge.columns"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### \"title\",  \"subject\" and \"date\" columns is not required for detecting the fake news, so I am going to drop the columns."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df_marge.drop([\"title\", \"subject\",\"date\"], axis = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "text     0\n",
       "class    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Randomly shuffling the dataframe "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.sample(frac = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text</th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>16928</th>\n",
       "      <td>It s almost as though we don t even have a Con...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9574</th>\n",
       "      <td>WASHINGTON (Reuters) - Members of Congress on ...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19541</th>\n",
       "      <td>We mentioned in an article yesterday that the ...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18391</th>\n",
       "      <td>Former President Barack Obama seems to be feel...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2409</th>\n",
       "      <td>WASHINGTON (Reuters) - Republicans on Sunday u...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                    text  class\n",
       "16928  It s almost as though we don t even have a Con...      0\n",
       "9574   WASHINGTON (Reuters) - Members of Congress on ...      1\n",
       "19541  We mentioned in an article yesterday that the ...      0\n",
       "18391  Former President Barack Obama seems to be feel...      0\n",
       "2409   WASHINGTON (Reuters) - Republicans on Sunday u...      1"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.reset_index(inplace = True)\n",
    "df.drop([\"index\"], axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['text', 'class'], dtype='object')"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>text</th>\n",
       "      <th>class</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>It s almost as though we don t even have a Con...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>WASHINGTON (Reuters) - Members of Congress on ...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>We mentioned in an article yesterday that the ...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Former President Barack Obama seems to be feel...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>WASHINGTON (Reuters) - Republicans on Sunday u...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                text  class\n",
       "0  It s almost as though we don t even have a Con...      0\n",
       "1  WASHINGTON (Reuters) - Members of Congress on ...      1\n",
       "2  We mentioned in an article yesterday that the ...      0\n",
       "3  Former President Barack Obama seems to be feel...      0\n",
       "4  WASHINGTON (Reuters) - Republicans on Sunday u...      1"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Creating a function to convert the text in lowercase, remove the extra space, special chr., ulr and links."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "def wordopt(text):\n",
    "    text = text.lower()\n",
    "    text = re.sub('\\[.*?\\]', '', text)\n",
    "    text = re.sub(\"\\\\W\",\" \",text) \n",
    "    text = re.sub('https?://\\S+|www\\.\\S+', '', text)\n",
    "    text = re.sub('<.*?>+', '', text)\n",
    "    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)\n",
    "    text = re.sub('\\n', '', text)\n",
    "    text = re.sub('\\w*\\d\\w*', '', text)    \n",
    "    return text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"text\"] = df[\"text\"].apply(wordopt)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Defining dependent and independent variable as x and y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = df[\"text\"]\n",
    "y = df[\"class\"]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Splitting the dataset into training set and testing set. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Convert text to vectors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "vectorization = TfidfVectorizer()\n",
    "xv_train = vectorization.fit_transform(x_train)\n",
    "xv_test = vectorization.transform(x_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "LR = LogisticRegression()\n",
    "LR.fit(xv_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_lr=LR.predict(xv_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9845879732739421"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "LR.score(xv_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.99      0.98      0.99      5918\n",
      "           1       0.98      0.99      0.98      5307\n",
      "\n",
      "    accuracy                           0.98     11225\n",
      "   macro avg       0.98      0.98      0.98     11225\n",
      "weighted avg       0.98      0.98      0.98     11225\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test, pred_lr))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Decision Tree Classification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier()"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "DT = DecisionTreeClassifier()\n",
    "DT.fit(xv_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_dt = DT.predict(xv_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9942984409799555"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "DT.score(xv_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.99      0.99      0.99      5918\n",
      "           1       0.99      0.99      0.99      5307\n",
      "\n",
      "    accuracy                           0.99     11225\n",
      "   macro avg       0.99      0.99      0.99     11225\n",
      "weighted avg       0.99      0.99      0.99     11225\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test, pred_dt))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Gradient Boosting Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import GradientBoostingClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "GradientBoostingClassifier(random_state=0)"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "GBC = GradientBoostingClassifier(random_state=0)\n",
    "GBC.fit(xv_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_gbc = GBC.predict(xv_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9947438752783965"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "GBC.score(xv_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       1.00      0.99      1.00      5918\n",
      "           1       0.99      1.00      0.99      5307\n",
      "\n",
      "    accuracy                           0.99     11225\n",
      "   macro avg       0.99      0.99      0.99     11225\n",
      "weighted avg       0.99      0.99      0.99     11225\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test, pred_gbc))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Random Forest Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(random_state=0)"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "RFC = RandomForestClassifier(random_state=0)\n",
    "RFC.fit(xv_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "pred_rfc = RFC.predict(xv_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9869042316258352"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "RFC.score(xv_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.99      0.99      0.99      5918\n",
      "           1       0.99      0.99      0.99      5307\n",
      "\n",
      "    accuracy                           0.99     11225\n",
      "   macro avg       0.99      0.99      0.99     11225\n",
      "weighted avg       0.99      0.99      0.99     11225\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test, pred_rfc))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model Testing With Manual Entry\n",
    "\n",
    "### News"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "def output_lable(n):\n",
    "    if n == 0:\n",
    "        return ' \"Fake News - (FALSE)\" '\n",
    "    elif n == 1:\n",
    "        return ' \"Not A Fake News - (TRUE)\" '\n",
    "    \n",
    "def manual_testing(news):\n",
    "    testing_news = {\"text\":[news]}\n",
    "    new_def_test = pd.DataFrame(testing_news)\n",
    "    new_def_test[\"text\"] = new_def_test[\"text\"].apply(wordopt) \n",
    "    new_x_test = new_def_test[\"text\"]\n",
    "    new_xv_test = vectorization.transform(new_x_test)\n",
    "    pred_LR = LR.predict(new_xv_test)\n",
    "    pred_DT = DT.predict(new_xv_test)\n",
    "    pred_GBC = GBC.predict(new_xv_test)\n",
    "    pred_RFC = RFC.predict(new_xv_test)\n",
    "\n",
    "    return print(\"\\n\\nLinear Regressior Prediction: {} \\nDecision Tree classifier Prediction: {} \\nGradient Boosting Classifier Prediction: {} \\nRandom Forest Classifier Prediction: {}\".format(output_lable(pred_LR[0]), \n",
    "                                                                                                              output_lable(pred_DT[0]), \n",
    "                                                                                                              output_lable(pred_GBC[0]), \n",
    "                                                                                                              output_lable(pred_RFC[0])))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyboardInterrupt",
     "evalue": "Interrupted by user",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-49-951b9af5e18d>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mnews\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mstr\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0minput\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[0mmanual_testing\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mnews\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mc:\\users\\mouizuddin\\appdata\\local\\programs\\python\\python39\\lib\\site-packages\\ipykernel\\kernelbase.py\u001b[0m in \u001b[0;36mraw_input\u001b[1;34m(self, prompt)\u001b[0m\n\u001b[0;32m    846\u001b[0m                 \u001b[1;34m\"raw_input was called, but this frontend does not support input requests.\"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    847\u001b[0m             )\n\u001b[1;32m--> 848\u001b[1;33m         return self._input_request(str(prompt),\n\u001b[0m\u001b[0;32m    849\u001b[0m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_parent_ident\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    850\u001b[0m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_parent_header\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mc:\\users\\mouizuddin\\appdata\\local\\programs\\python\\python39\\lib\\site-packages\\ipykernel\\kernelbase.py\u001b[0m in \u001b[0;36m_input_request\u001b[1;34m(self, prompt, ident, parent, password)\u001b[0m\n\u001b[0;32m    890\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mKeyboardInterrupt\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    891\u001b[0m                 \u001b[1;31m# re-raise KeyboardInterrupt, to truncate traceback\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 892\u001b[1;33m                 \u001b[1;32mraise\u001b[0m \u001b[0mKeyboardInterrupt\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Interrupted by user\"\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    893\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mException\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    894\u001b[0m                 \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mlog\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mwarning\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Invalid Message:\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mexc_info\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m: Interrupted by user"
     ]
    }
   ],
   "source": [
    "news = str(input())\n",
    "manual_testing(news) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Time complexities \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Random forest classfiers : O(d∗n∗log(n)) - O(log(n))\n",
    "- DTC : O(m · n2) - O(log(n^2)\n",
    "- LR : O(n)\n",
    "- O(n*logn) > O(n)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
