{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "5aaf4d35-022a-4848-997e-89f5d15950ae",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Hello World\n"
     ]
    }
   ],
   "source": [
    "print('Hello World')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "93c7e5d3-3ed6-4997-b30b-2c5af59954e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from torch import nn\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import cv2\n",
    "import torchvision"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "id": "0597a8de-7a65-4750-956a-a1904d15c601",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'cuda'"
      ]
     },
     "execution_count": 122,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "device"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "e029d86e-9b31-4919-aa8c-ed732e629341",
   "metadata": {},
   "outputs": [],
   "source": [
    "LABEL_MAP = {\n",
    "    'none': 0, 'Center': 1, 'Donut': 2, 'Edge-Loc': 3, \n",
    "    'Edge-Ring': 4, 'Loc': 5, 'Random': 6, 'Scratch': 7, 'Near-full': 8\n",
    "}\n",
    "\n",
    "NUM_MAP = {\n",
    "    0: 'none', 1: 'Center', 2: 'Donut', 3: 'Edge-Loc', \n",
    "    4: 'Edge-Ring', 5: 'Loc', 6: 'Random', 7: 'Scratch', 8: 'Near-full'\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "d3790a59-1a56-4212-b2a7-efa5a5b22e21",
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
       "      <th>waferMap</th>\n",
       "      <th>dieSize</th>\n",
       "      <th>lotName</th>\n",
       "      <th>waferIndex</th>\n",
       "      <th>trianTestLabel</th>\n",
       "      <th>failureType</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...</td>\n",
       "      <td>1683.0</td>\n",
       "      <td>lot1</td>\n",
       "      <td>1.0</td>\n",
       "      <td>[[Training]]</td>\n",
       "      <td>[[none]]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...</td>\n",
       "      <td>1683.0</td>\n",
       "      <td>lot1</td>\n",
       "      <td>2.0</td>\n",
       "      <td>[[Training]]</td>\n",
       "      <td>[[none]]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...</td>\n",
       "      <td>1683.0</td>\n",
       "      <td>lot1</td>\n",
       "      <td>3.0</td>\n",
       "      <td>[[Training]]</td>\n",
       "      <td>[[none]]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...</td>\n",
       "      <td>1683.0</td>\n",
       "      <td>lot1</td>\n",
       "      <td>4.0</td>\n",
       "      <td>[[Training]]</td>\n",
       "      <td>[[none]]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...</td>\n",
       "      <td>1683.0</td>\n",
       "      <td>lot1</td>\n",
       "      <td>5.0</td>\n",
       "      <td>[[Training]]</td>\n",
       "      <td>[[none]]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>811452</th>\n",
       "      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1,...</td>\n",
       "      <td>600.0</td>\n",
       "      <td>lot47542</td>\n",
       "      <td>23.0</td>\n",
       "      <td>[[Test]]</td>\n",
       "      <td>[[Edge-Ring]]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>811453</th>\n",
       "      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1,...</td>\n",
       "      <td>600.0</td>\n",
       "      <td>lot47542</td>\n",
       "      <td>24.0</td>\n",
       "      <td>[[Test]]</td>\n",
       "      <td>[[Edge-Loc]]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>811454</th>\n",
       "      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1,...</td>\n",
       "      <td>600.0</td>\n",
       "      <td>lot47542</td>\n",
       "      <td>25.0</td>\n",
       "      <td>[[Test]]</td>\n",
       "      <td>[[Edge-Ring]]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>811455</th>\n",
       "      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,...</td>\n",
       "      <td>600.0</td>\n",
       "      <td>lot47543</td>\n",
       "      <td>1.0</td>\n",
       "      <td>[]</td>\n",
       "      <td>[]</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>811456</th>\n",
       "      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1,...</td>\n",
       "      <td>600.0</td>\n",
       "      <td>lot47543</td>\n",
       "      <td>2.0</td>\n",
       "      <td>[]</td>\n",
       "      <td>[]</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>811457 rows × 6 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                 waferMap  dieSize   lotName  \\\n",
       "0       [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...   1683.0      lot1   \n",
       "1       [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...   1683.0      lot1   \n",
       "2       [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...   1683.0      lot1   \n",
       "3       [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...   1683.0      lot1   \n",
       "4       [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...   1683.0      lot1   \n",
       "...                                                   ...      ...       ...   \n",
       "811452  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1,...    600.0  lot47542   \n",
       "811453  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1,...    600.0  lot47542   \n",
       "811454  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1,...    600.0  lot47542   \n",
       "811455  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,...    600.0  lot47543   \n",
       "811456  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1,...    600.0  lot47543   \n",
       "\n",
       "        waferIndex trianTestLabel    failureType  \n",
       "0              1.0   [[Training]]       [[none]]  \n",
       "1              2.0   [[Training]]       [[none]]  \n",
       "2              3.0   [[Training]]       [[none]]  \n",
       "3              4.0   [[Training]]       [[none]]  \n",
       "4              5.0   [[Training]]       [[none]]  \n",
       "...            ...            ...            ...  \n",
       "811452        23.0       [[Test]]  [[Edge-Ring]]  \n",
       "811453        24.0       [[Test]]   [[Edge-Loc]]  \n",
       "811454        25.0       [[Test]]  [[Edge-Ring]]  \n",
       "811455         1.0             []             []  \n",
       "811456         2.0             []             []  \n",
       "\n",
       "[811457 rows x 6 columns]"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataframe = pd.read_pickle(r\"archive\\LSWMD.pkl\")\n",
    "dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "f3a1dbd2-a41f-4c40-bc30-cecdcb2098b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "def remove_null(df):\n",
    "    df = df[df['failureType'].notnull()].reset_index(drop=True)\n",
    "    \n",
    "    def extract_label_safely(x):\n",
    "        # Check if it's a numpy array\n",
    "        if isinstance(x, np.ndarray):\n",
    "            # CRITICAL: Verify the array actually has elements before indexing\n",
    "            if x.size > 0 and len(x[0]) > 0:\n",
    "                return x[0][0]\n",
    "            else:\n",
    "                return \"unlabeled\" # Mark empty arrays for dropping\n",
    "        elif isinstance(x, str):\n",
    "            return x\n",
    "        return \"unlabeled\"\n",
    "\n",
    "    \n",
    "    \n",
    "    \n",
    "    df['clean_label'] = df['failureType'].apply(extract_label_safely)\n",
    "    df = df[df['clean_label'] != \"unlabeled\"].reset_index(drop=True)\n",
    "    df = df.drop(['dieSize', 'lotName', 'waferIndex', 'trianTestLabel', 'failureType'], axis=1)  \n",
    "\n",
    "    return df\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "0b2c0ac8-8c61-49e3-b0df-31c368f75f53",
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
       "      <th>waferMap</th>\n",
       "      <th>clean_label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>172945</th>\n",
       "      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2,...</td>\n",
       "      <td>Edge-Loc</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>172946</th>\n",
       "      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,...</td>\n",
       "      <td>Edge-Loc</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>172947</th>\n",
       "      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1,...</td>\n",
       "      <td>Edge-Ring</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>172948</th>\n",
       "      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1,...</td>\n",
       "      <td>Edge-Loc</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>172949</th>\n",
       "      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1,...</td>\n",
       "      <td>Edge-Ring</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                 waferMap clean_label\n",
       "172945  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2,...    Edge-Loc\n",
       "172946  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,...    Edge-Loc\n",
       "172947  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1,...   Edge-Ring\n",
       "172948  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1,...    Edge-Loc\n",
       "172949  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1,...   Edge-Ring"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cleaned_df = remove_null(dataframe)\n",
    "cleaned_df.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "56651386-02db-46c2-9292-e3e897bcdf9e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "147431\n",
      "5189\n",
      "9680\n"
     ]
    }
   ],
   "source": [
    "print(len(cleaned_df[cleaned_df['clean_label'] == 'none']))\n",
    "print(len(cleaned_df[cleaned_df['clean_label'] == 'Edge-Loc']))\n",
    "print(len(cleaned_df[cleaned_df['clean_label'] == 'Edge-Ring']))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "50ef8c77-1a48-4629-9c91-410881e42a45",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "samples : 172950\n"
     ]
    }
   ],
   "source": [
    "print(\"samples : \" + str(len(cleaned_df)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "ca9fc9ea-8d21-49d7-836f-d59fd409b2ba",
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
       "      <th>waferMap</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>172945</th>\n",
       "      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2,...</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>172946</th>\n",
       "      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,...</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>172947</th>\n",
       "      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1,...</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>172948</th>\n",
       "      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1,...</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>172949</th>\n",
       "      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1,...</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                 waferMap  label\n",
       "172945  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2,...      3\n",
       "172946  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,...      3\n",
       "172947  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1,...      4\n",
       "172948  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1,...      3\n",
       "172949  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1,...      4"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def hot_encode(df, labels):\n",
    "    def apply_labels(x):\n",
    "        return labels[x]\n",
    "    df['label'] = df['clean_label'].apply(apply_labels)\n",
    "    df = df.drop('clean_label', axis=1)\n",
    "    return df\n",
    "\n",
    "final_data = hot_encode(cleaned_df, LABEL_MAP)\n",
    "final_data.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "07bda69a-1d0b-449f-90f3-8de8de3f3467",
   "metadata": {},
   "outputs": [],
   "source": [
    "def find_weights(df):\n",
    "    weights = []\n",
    "    total_samples = len(df)\n",
    "    num_classes = len(LABEL_MAP)\n",
    "    \n",
    "    # 1. Loop through each class index\n",
    "    for i in range(num_classes):\n",
    "        num = len(df[df['label'] == i])\n",
    "        \n",
    "        # Calculate standard inverse-frequency weight\n",
    "        # Formula: Total Samples / (Number of Classes * Class Count)\n",
    "        weight = total_samples / (num_classes * num) if num > 0 else 0.0\n",
    "        weights.append(weight)\n",
    "        \n",
    "        # Print out both the sample count and calculated weight for scannability\n",
    "        print(f\"Label  {i} :  Count = {num:<8} | Weight = {weight:.4f}\")\n",
    "        \n",
    "    return np.array(weights)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "db5f19e7-d434-46bb-b6b9-343ed9266fb5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Label  0 :  Count = 147431   | Weight = 0.1303\n",
      "Label  1 :  Count = 4294     | Weight = 4.4752\n",
      "Label  2 :  Count = 555      | Weight = 34.6246\n",
      "Label  3 :  Count = 5189     | Weight = 3.7033\n",
      "Label  4 :  Count = 9680     | Weight = 1.9852\n",
      "Label  5 :  Count = 3593     | Weight = 5.3484\n",
      "Label  6 :  Count = 866      | Weight = 22.1901\n",
      "Label  7 :  Count = 1193     | Weight = 16.1079\n",
      "Label  8 :  Count = 149      | Weight = 128.9709\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([  0.13034346,   4.47523676,  34.62462462,   3.70334682,\n",
       "         1.98519284,   5.34836256,  22.19014627,  16.10785136,\n",
       "       128.97091723])"
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "find_weights(final_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "06e13dab-8795-4cf9-a3d8-0359feaee413",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Label  0 :  Count = 10000    | Weight = 0.3947\n",
      "Label  1 :  Count = 4294     | Weight = 0.9191\n",
      "Label  2 :  Count = 555      | Weight = 7.1109\n",
      "Label  3 :  Count = 5189     | Weight = 0.7606\n",
      "Label  4 :  Count = 9680     | Weight = 0.4077\n",
      "Label  5 :  Count = 3593     | Weight = 1.0984\n",
      "Label  6 :  Count = 866      | Weight = 4.5572\n",
      "Label  7 :  Count = 1193     | Weight = 3.3081\n",
      "Label  8 :  Count = 149      | Weight = 26.4870\n",
      "[ 0.39465556  0.91908606  7.11091091  0.76056187  0.40770202  1.09840121\n",
      "  4.55722351  3.30809351 26.48695004]\n"
     ]
    }
   ],
   "source": [
    "def balance_dataset(df):\n",
    "    # 1. Downsample label 0 to 10,000 rows\n",
    "    df_normal = df[df['label'] == 0].sample(n=10000, random_state=42)\n",
    "    \n",
    "    # 2. Keep all other defect labels\n",
    "    df_defects = df[df['label'] != 0]\n",
    "    \n",
    "    # 3. Combine and shuffle the rows completely\n",
    "    balanced_df = pd.concat([df_normal, df_defects]).sample(frac=1, random_state=42).reset_index(drop=True)\n",
    "    \n",
    "    return balanced_df\n",
    "\n",
    "bala = balance_dataset(final_data)\n",
    "LABEL_WEIGHTS = find_weights(bala)\n",
    "print(LABEL_WEIGHTS)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "id": "32154602-d04d-422d-b47b-f94fa1cc84b7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Edge-Loc')"
      ]
     },
     "execution_count": 97,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "n = 856\n",
    "\n",
    "plt.imshow(bala['waferMap'][n], cmap='gray')\n",
    "plt.title(NUM_MAP[bala['label'][n]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "id": "aba4a437-bc0d-44e1-921a-f5b086bdf42f",
   "metadata": {},
   "outputs": [],
   "source": [
    "from torch.utils.data import TensorDataset\n",
    "from torchvision.transforms import functional as F\n",
    "from torchvision.transforms import InterpolationMode\n",
    "\n",
    "def process_dataframe_to_tensors(df, target_size=(100, 100)):\n",
    "    \"\"\"\n",
    "    Processes a Pandas DataFrame containing variable-sized wafer maps \n",
    "    and returns a unified PyTorch TensorDataset.\n",
    "    \n",
    "    Parameters:\n",
    "    df (pd.DataFrame): The input dataframe containing 'waferMap' and 'label' columns.\n",
    "    target_size (tuple): The target spatial dimension (Height, Width) for the output tensors.\n",
    "    \n",
    "    Returns:\n",
    "    torch.utils.data.TensorDataset: A dataset containing the stacked, normalized \n",
    "                                    images and their corresponding integer labels.\n",
    "    \"\"\"\n",
    "    processed_images = []\n",
    "    \n",
    "    print(f\"Starting tensor conversion for {len(df)} records...\")\n",
    "    \n",
    "    for idx, row in df.iterrows():\n",
    "        raw_matrix = row['waferMap']\n",
    "        \n",
    "        # 1. Convert raw numpy array to a PyTorch FloatTensor \n",
    "        # Shape transition: (H, W) -> (1, H, W) to add the expected channel dimension\n",
    "        tensor_map = torch.tensor(raw_matrix, dtype=torch.float32).unsqueeze(0)\n",
    "        \n",
    "        # 2. Resize using NEAREST to preserve discrete categorical values (0, 1, 2)\n",
    "        resized_tensor = F.resize(\n",
    "            tensor_map, \n",
    "            size=target_size, \n",
    "            interpolation=InterpolationMode.NEAREST\n",
    "        )\n",
    "        \n",
    "        # 3. Scale categorical values [0, 1, 2] to an optimal range of [0.0, 0.5, 1.0]\n",
    "        normalized_tensor = resized_tensor / 2.0\n",
    "        \n",
    "        processed_images.append(normalized_tensor)\n",
    "        \n",
    "    # Stack the list of individual (1, 100, 100) tensors into a giant (N, 1, 100, 100) contiguous tensor block\n",
    "    X_tensor = torch.stack(processed_images)\n",
    "    \n",
    "    # Convert the 'label' column directly into a contiguous LongTensor\n",
    "    y_tensor = torch.tensor(df['label'].values, dtype=torch.long)\n",
    "    \n",
    "    print(f\"Successfully processed tensor shapes -> X: {X_tensor.shape} | y: {y_tensor.shape}\")\n",
    "    \n",
    "    # Wrap cleanly into a native PyTorch TensorDataset\n",
    "    return TensorDataset(X_tensor, y_tensor)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "id": "a0d6f30a-5cdb-46fe-828c-b92f27fb7304",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Starting tensor conversion for 35519 records...\n",
      "Successfully processed tensor shapes -> X: torch.Size([35519, 1, 100, 100]) | y: torch.Size([35519])\n"
     ]
    }
   ],
   "source": [
    "dataset = process_dataframe_to_tensors(bala)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "id": "33e9427f-1c0b-4c09-944a-72135806170d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAaEAAAGxCAYAAADLfglZAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjYsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvq6yFwwAAAAlwSFlzAAAPYQAAD2EBqD+naQAAIElJREFUeJzt3X9wVNX9xvFnIbAkmGwByy5BwDATBxQtCEgFKjgIWsDR0qr8EFQ6HWhAibRCUqysVBJILcOoFQt1ApZSGCtF1GqJoKkIrRGLUqBgxxTjjzRV6S4qSYSc7x+U/boGNJts8tlN3q+Z+8eee+7ec89d8nA2N+d4nHNOAAAYaGfdAABA20UIAQDMEEIAADOEEADADCEEADBDCAEAzBBCAAAzhBAAwAwhBAAwQwihzVm7dq08Hk9kS0lJUY8ePTR58mS9+eabJm0KBoPyeDwm5wYspVg3ALBSXFysfv36qbq6Wi+//LKWLl2qF154Qf/4xz/UpUsX6+YBbQIhhDZrwIABGjJkiCRp9OjROnnypBYvXqwtW7botttuM24d0DbwdRzwP6cD6d///rckqbq6Wj/60Y80cOBA+Xw+de3aVZdffrmefPLJesd6PB7NnTtXv/nNb9S/f3+lpaXpG9/4hp5++ul6dZ955hkNHDhQXq9XWVlZuv/++8/YnurqauXn5ysrK0sdO3ZUz549NWfOHP33v/+Nqnf++edr4sSJevrppzVo0CClpqaqf//+kXOvXbtW/fv3V+fOnXXZZZfp1VdfbUo3AXHFSAj4n/LycknSBRdcIEmqqanRRx99pB//+Mfq2bOnamtr9fzzz2vSpEkqLi7WjBkzoo5/5plnVFZWpiVLluicc85RUVGRvvOd7+jQoUPq27evJGn79u267rrrdPnll2vjxo06efKkioqKIsF3mnNO119/vbZv3678/Hx961vf0htvvKHFixdr9+7d2r17t7xeb6T+66+/rvz8fC1atEg+n0/33nuvJk2apPz8fG3fvl0FBQXyeDxauHChJk6cqPLycqWmpjZndwIN44A2pri42Elyf/nLX9xnn33mjh075p577jkXCATcFVdc4T777LMzHnfixAn32Wefue9///tu0KBBUfskOb/f78LhcKSssrLStWvXzhUWFkbKhg0b5jIzM93x48cjZeFw2HXt2tV9/p/jc8895yS5oqKiqPNs2rTJSXKrV6+OlPXp08elpqa6d955J1K2d+9eJ8n16NHDffLJJ5HyLVu2OElu69atDe0uoFnxdRzarG9+85vq0KGD0tPTdc0116hLly568sknlZLy/18QPP744xoxYoTOOeccpaSkqEOHDnr00Ud18ODBeu935ZVXKj09PfLa7/ere/fuOnLkiCTpk08+UVlZmSZNmqROnTpF6qWnp+vaa6+Neq8dO3ZIkm699dao8htuuEGdO3fW9u3bo8oHDhyonj17Rl73799f0qnfdaWlpdUrP90mwBohhDbrscceU1lZmXbs2KFZs2bp4MGDmjJlSmT/5s2bdeONN6pnz55av369du/erbKyMs2cOVPV1dX13q9bt271yrxer44fPy5JOnr0qOrq6hQIBOrV+2LZhx9+qJSUFH3961+PKvd4PAoEAvrwww+jyrt27Rr1umPHjl9afqb2Axb4nRDarP79+0ceRrjyyit18uRJ/frXv9bvf/97fe9739P69euVlZWlTZs2Rf0NT01NTaPO16VLF3k8HlVWVtbb98Wybt266cSJE/rPf/4TFUTOOVVWVmro0KGNagOQaBgJAf9TVFSkLl266J577lFdXZ08Ho86duwYFUCVlZVnfDquIU4/nbZ58+aokcixY8f01FNPRdUdM2aMJGn9+vVR5U888YQ++eSTyH4g2RFCwP906dJF+fn5OnjwoDZs2KCJEyfq0KFDysnJ0Y4dO7Ru3TqNHDlSPXr0aPQ5fvazn6myslJjx47Vli1b9MQTT2jMmDHq3LlzVL2xY8fq6quv1sKFC3Xvvffq+eef14oVK3Tbbbdp0KBBmj59elMvF0gIhBDwObfffrt69+6tJUuWaMaMGVq2bJmeffZZjR8/XsuXL1deXp6mTp3a6Pc/HT7hcFg33XST5s+fr+9+97uaOXNmVD2Px6MtW7Zo/vz5Ki4u1vjx43X//fdr+vTp2rFjR9Tj2UAy8zjnnHUjAABtEyMhAIAZQggAYIYQAgCYIYQAAGYIIQCAmWYLoYcfflhZWVnq1KmTBg8erJdeeqm5TgUASFLNMm3Ppk2blJubq4cfflgjRozQr371K33729/WgQMH1Lt37y89tq6uTu+9957S09NZ7hgAkpBzTseOHVNmZqbatfuKsU5zTM192WWXudmzZ0eV9evXz+Xl5X3lsRUVFU4S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+/btczfddJPr0aOHC4fDhi2388orr7jzzz/fXXLJJW7evHmRcvrJuY8++sj16dPH3Xrrre6vf/2rKy8vd88//7z75z//GalDPzl33333uW7durmnn37alZeXu8cff9ydc845buXKlZE69FPTJWQIXXbZZW727NlRZf369XN5eXlGLUosVVVVTpIrLS11zjlXV1fnAoGAW7ZsWaROdXW18/l87pFHHrFqppljx4657OxsV1JS4kaNGhUJIfrplIULF7qRI0eedT/9dMqECRPczJkzo8omTZrkbr75Zucc/RQvCfd1XG1trfbs2aNx48ZFlY8bN067du0yalViCYVCkqSuXbtKksrLy1VZWRnVZ16vV6NGjWqTfTZnzhxNmDBBV111VVQ5/XTK1q1bNWTIEN1www3q3r27Bg0apDVr1kT200+njBw5Utu3b9fhw4clSa+//rp27typ8ePHS6Kf4iXhZtH+4IMPdPLkSfn9/qhyv9+vyspKo1YlDuec5s+fr5EjR2rAgAGSFOmXM/XZkSNHWryNljZu3KjXXntNZWVl9fbRT6e89dZbWrVqlebPn6+f/OQneuWVV3THHXfI6/VqxowZ9NP/LFy4UKFQSP369VP79u118uRJLV26VFOmTJHE5yleEi6ETvN4PFGvnXP1ytqiuXPn6o033tDOnTvr7WvrfVZRUaF58+Zp27Zt6tSp01nrtfV+qqur05AhQ1RQUCBJGjRokPbv369Vq1ZpxowZkXptvZ82bdqk9evXa8OGDbrooou0d+9e5ebmKjMzU7fcckukXlvvp6ZKuK/jzj33XLVv377eqKeqqqre/zjamttvv11bt27VCy+8ELVaYSAQkKQ232d79uxRVVWVBg8erJSUFKWkpKi0tFQPPPCAUlJSIn3R1vupR48euvDCC6PK+vfvr7ffflsSn6fT7rrrLuXl5Wny5Mm6+OKLNX36dN15550qLCyURD/FS8KFUMeOHTV48GCVlJRElZeUlGj48OFGrbLlnNPcuXO1efNm7dixQ1lZWVH7s7KyFAgEovqstrZWpaWlbarPxowZo3379mnv3r2RbciQIZo2bZr27t2rvn370k+SRowYUe8R/8OHD6tPnz6S+Dyd9umnn9ZbFbR9+/aRR7TppzgxfCjirE4/ov3oo4+6AwcOuNzcXNe5c2f3r3/9y7ppJn74wx86n8/nXnzxRff+++9Htk8//TRSZ9myZc7n87nNmze7ffv2uSlTpvCoqHNRT8c5Rz85d+rx9ZSUFLd06VL35ptvut/+9rcuLS3NrV+/PlKHfnLulltucT179ow8or1582Z37rnnugULFkTq0E9Nl5Ah5Jxzv/zlL12fPn1cx44d3aWXXhp5HLktknTGrbi4OFKnrq7OLV682AUCAef1et0VV1zh9u3bZ9foBPHFEKKfTnnqqafcgAEDnNfrdf369XOrV6+O2k8/ORcOh928efNc7969XadOnVzfvn3dokWLXE1NTaQO/dR0rCcEADCTcL8TAgC0HYQQAMAMIQQAMEMIAQDMEEIAADOEEADADCEEADBDCAEAzBBCAAAzhBAAwAwhBAAw83+VD+YVk0VKAAAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "n=120\n",
    "\n",
    "plt.imshow(dataset[n][0].permute(1, 2, 0), cmap='gray')\n",
    "plt.title(NUM_MAP[dataset[n][1].item()])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "id": "c7f9b24e-2c91-495a-9a38-0b619fea94fd",
   "metadata": {},
   "outputs": [],
   "source": [
    "from torchvision.transforms import v2\n",
    "from torchvision.transforms import InterpolationMode\n",
    "\n",
    "# Training transforms: Only apply discrete spatial flips and rotations\n",
    "train_transform = v2.Compose([\n",
    "    v2.RandomHorizontalFlip(p=0.5),\n",
    "    v2.RandomVerticalFlip(p=0.5),\n",
    "    v2.RandomRotation(degrees=(0, 270), interpolation=InterpolationMode.NEAREST)\n",
    "])\n",
    "\n",
    "val_transform = v2.Identity()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "id": "2ccd073a-9c44-4783-b27d-9359d2fbb4a0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Data flow pipeline completely sealed and ready for optimization!\n"
     ]
    }
   ],
   "source": [
    "from torch.utils.data import random_split, DataLoader\n",
    "from torch.utils.data import Dataset\n",
    "\n",
    "class V2AugmentedDataset(Dataset):\n",
    "    def __init__(self, tensor_dataset, transform):\n",
    "        \"\"\"Wraps your pre-processed TensorDataset for rapid V2 transformations.\"\"\"\n",
    "        self.tensors = tensor_dataset.tensors\n",
    "        self.X = self.tensors[0]  # Your contiguous (N, 1, 100, 100) block\n",
    "        self.y = self.tensors[1]  # Your contiguous (N,) label block\n",
    "        self.transform = transform\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.X)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        img = self.X[idx]\n",
    "        label = self.y[idx]\n",
    "        \n",
    "        # Execute rapid on-the-fly geometric shifts\n",
    "        img = self.transform(img)\n",
    "        \n",
    "        return img, label\n",
    "\n",
    "# Assuming your function output is saved as `wafer_dataset`\n",
    "total_samples = len(dataset)\n",
    "train_size = int(0.8 * total_samples)\n",
    "val_size = total_samples - train_size\n",
    "\n",
    "# 1. Split your core dataset using a fixed seed for reproducibility\n",
    "raw_train_split, raw_val_split = random_split(\n",
    "    dataset, \n",
    "    [train_size, val_size], \n",
    "    generator=torch.Generator().manual_seed(42)\n",
    ")\n",
    "\n",
    "# 2. Attach the wrappers\n",
    "train_dataset = V2AugmentedDataset(raw_train_split.dataset, transform=train_transform)\n",
    "val_dataset = V2AugmentedDataset(raw_val_split.dataset, transform=val_transform)\n",
    "\n",
    "# 3. Spin up high-speed parallel workers\n",
    "train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)\n",
    "val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)\n",
    "\n",
    "print(\"✅ Data flow pipeline completely sealed and ready for optimization!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "id": "f279f7c8-3645-450f-8315-b7f3bcc2b203",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "278"
      ]
     },
     "execution_count": 115,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(train_loader)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0be01077-3ef5-494b-9a3a-a3c85be4e6cf",
   "metadata": {},
   "source": [
    "MODEL MAKING"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "id": "d3ffdcb7-5933-4cd4-968e-1158c74d4e6a",
   "metadata": {},
   "outputs": [],
   "source": [
    "class BasicBlock_V0(nn.Module):\n",
    "    def __init__(self, in_channels, out_channels,  stride=1, downsample=None):\n",
    "        super().__init__()\n",
    "        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)\n",
    "        self.bn1 = nn.BatchNorm2d(out_channels)\n",
    "\n",
    "        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)\n",
    "        self.bn2 = nn.BatchNorm2d(out_channels)\n",
    "        self.relu = nn.ReLU()\n",
    "        self.downsample = downsample\n",
    "\n",
    "    def forward(self, x):\n",
    "        ident = x\n",
    "        out = self.conv1(x)\n",
    "        out = self.bn1(out)\n",
    "        out = self.relu(out)\n",
    "\n",
    "        out = self.conv2(out)\n",
    "        out = self.bn2(out)\n",
    "        if self.downsample is not None:\n",
    "            ident = self.downsample(ident)\n",
    "        out = out + ident\n",
    "        out = self.relu(out)\n",
    "\n",
    "        return out\n",
    "\n",
    "class ResNet_Wafer_V0(nn.Module):\n",
    "    def __init__(self, num_classes=9):\n",
    "        super().__init__()\n",
    "        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)\n",
    "        self.bn1 = nn.BatchNorm2d(64)\n",
    "        self.relu = nn.ReLU()\n",
    "        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)\n",
    "\n",
    "        self.layer1 = self.make_skip_layer(64, 64, 2, 1)\n",
    "        self.layer2 = self.make_skip_layer(64, 128, 2, 2)\n",
    "        self.layer3 = self.make_skip_layer(128, 256, 2, 2)\n",
    "        self.layer4 = self.make_skip_layer(256, 512, 2, 2)\n",
    "\n",
    "        self.avgpool = nn.AdaptiveAvgPool2d((1,1))\n",
    "        self.line = nn.Linear(512, num_classes)\n",
    "\n",
    "        self.flatten = nn.Flatten()\n",
    "        \n",
    "\n",
    "\n",
    "    def make_skip_layer(self, in_channels, out_channels, blocks, stride=1):\n",
    "        downsample = None\n",
    "\n",
    "        if stride!=1 or (in_channels!=out_channels):\n",
    "            downsample = nn.Sequential(\n",
    "                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),\n",
    "                nn.BatchNorm2d(out_channels)\n",
    "            )\n",
    "        layers = []\n",
    "        layers.append(BasicBlock_V0(in_channels, out_channels, stride, downsample))\n",
    "        for i in range(1, blocks):\n",
    "            layers.append(BasicBlock_V0(out_channels, out_channels))\n",
    "\n",
    "        return nn.Sequential(*layers)\n",
    "\n",
    "    def forward(self, x):\n",
    "        out = self.conv1(x)\n",
    "        out = self.bn1(out)\n",
    "        out = self.relu(out)\n",
    "        out = self.maxpool1(out)\n",
    "\n",
    "        out = self.layer1(out)\n",
    "        out = self.layer2(out)\n",
    "        out = self.layer3(out)\n",
    "        out = self.layer4(out)\n",
    "\n",
    "        out = self.avgpool(out)\n",
    "        out = self.flatten(out)\n",
    "        out = self.line(out)\n",
    "\n",
    "        return out"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "id": "1bbfef6e-0258-4e8d-ac2e-67f82f58cf66",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ResNet_Wafer_V0(\n",
       "  (conv1): Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)\n",
       "  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "  (relu): ReLU()\n",
       "  (maxpool1): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)\n",
       "  (layer1): Sequential(\n",
       "    (0): BasicBlock_V0(\n",
       "      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU()\n",
       "    )\n",
       "    (1): BasicBlock_V0(\n",
       "      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU()\n",
       "    )\n",
       "  )\n",
       "  (layer2): Sequential(\n",
       "    (0): BasicBlock_V0(\n",
       "      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU()\n",
       "      (downsample): Sequential(\n",
       "        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)\n",
       "        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      )\n",
       "    )\n",
       "    (1): BasicBlock_V0(\n",
       "      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU()\n",
       "    )\n",
       "  )\n",
       "  (layer3): Sequential(\n",
       "    (0): BasicBlock_V0(\n",
       "      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU()\n",
       "      (downsample): Sequential(\n",
       "        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)\n",
       "        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      )\n",
       "    )\n",
       "    (1): BasicBlock_V0(\n",
       "      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU()\n",
       "    )\n",
       "  )\n",
       "  (layer4): Sequential(\n",
       "    (0): BasicBlock_V0(\n",
       "      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU()\n",
       "      (downsample): Sequential(\n",
       "        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)\n",
       "        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      )\n",
       "    )\n",
       "    (1): BasicBlock_V0(\n",
       "      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)\n",
       "      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n",
       "      (relu): ReLU()\n",
       "    )\n",
       "  )\n",
       "  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))\n",
       "  (line): Linear(in_features=512, out_features=9, bias=True)\n",
       "  (flatten): Flatten(start_dim=1, end_dim=-1)\n",
       ")"
      ]
     },
     "execution_count": 128,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "resnet_model_0 = ResNet_Wafer_V0().to(device)\n",
    "resnet_model_0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "id": "ec8a2175-0bc8-403e-9bd9-c0d6b2351f70",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total Parameters:     11,174,857\n",
      "Trainable Parameters: 11,174,857\n"
     ]
    }
   ],
   "source": [
    "# 1. Calculate every parameter in the architecture graph\n",
    "total_params = sum(p.numel() for p in resnet_model_0.parameters())\n",
    "\n",
    "# 2. Calculate only the active weights updating during backpropagation\n",
    "trainable_params = sum(p.numel() for p in resnet_model_0.parameters() if p.requires_grad)\n",
    "\n",
    "print(f\"Total Parameters:     {total_params:,}\")\n",
    "print(f\"Trainable Parameters: {trainable_params:,}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "id": "45972ad4-b954-4b8d-ab8b-53c99cfc4c9e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "==========================================================================================\n",
       "Layer (type:depth-idx)                   Output Shape              Param #\n",
       "==========================================================================================\n",
       "ResNet_Wafer_V0                          [128, 9]                  --\n",
       "├─Conv2d: 1-1                            [128, 64, 50, 50]         3,136\n",
       "├─BatchNorm2d: 1-2                       [128, 64, 50, 50]         128\n",
       "├─ReLU: 1-3                              [128, 64, 50, 50]         --\n",
       "├─MaxPool2d: 1-4                         [128, 64, 25, 25]         --\n",
       "├─Sequential: 1-5                        [128, 64, 25, 25]         --\n",
       "│    └─BasicBlock_V0: 2-1                [128, 64, 25, 25]         --\n",
       "│    │    └─Conv2d: 3-1                  [128, 64, 25, 25]         36,864\n",
       "│    │    └─BatchNorm2d: 3-2             [128, 64, 25, 25]         128\n",
       "│    │    └─ReLU: 3-3                    [128, 64, 25, 25]         --\n",
       "│    │    └─Conv2d: 3-4                  [128, 64, 25, 25]         36,864\n",
       "│    │    └─BatchNorm2d: 3-5             [128, 64, 25, 25]         128\n",
       "│    │    └─ReLU: 3-6                    [128, 64, 25, 25]         --\n",
       "│    └─BasicBlock_V0: 2-2                [128, 64, 25, 25]         --\n",
       "│    │    └─Conv2d: 3-7                  [128, 64, 25, 25]         36,864\n",
       "│    │    └─BatchNorm2d: 3-8             [128, 64, 25, 25]         128\n",
       "│    │    └─ReLU: 3-9                    [128, 64, 25, 25]         --\n",
       "│    │    └─Conv2d: 3-10                 [128, 64, 25, 25]         36,864\n",
       "│    │    └─BatchNorm2d: 3-11            [128, 64, 25, 25]         128\n",
       "│    │    └─ReLU: 3-12                   [128, 64, 25, 25]         --\n",
       "├─Sequential: 1-6                        [128, 128, 13, 13]        --\n",
       "│    └─BasicBlock_V0: 2-3                [128, 128, 13, 13]        --\n",
       "│    │    └─Conv2d: 3-13                 [128, 128, 13, 13]        73,728\n",
       "│    │    └─BatchNorm2d: 3-14            [128, 128, 13, 13]        256\n",
       "│    │    └─ReLU: 3-15                   [128, 128, 13, 13]        --\n",
       "│    │    └─Conv2d: 3-16                 [128, 128, 13, 13]        147,456\n",
       "│    │    └─BatchNorm2d: 3-17            [128, 128, 13, 13]        256\n",
       "│    │    └─Sequential: 3-18             [128, 128, 13, 13]        8,448\n",
       "│    │    └─ReLU: 3-19                   [128, 128, 13, 13]        --\n",
       "│    └─BasicBlock_V0: 2-4                [128, 128, 13, 13]        --\n",
       "│    │    └─Conv2d: 3-20                 [128, 128, 13, 13]        147,456\n",
       "│    │    └─BatchNorm2d: 3-21            [128, 128, 13, 13]        256\n",
       "│    │    └─ReLU: 3-22                   [128, 128, 13, 13]        --\n",
       "│    │    └─Conv2d: 3-23                 [128, 128, 13, 13]        147,456\n",
       "│    │    └─BatchNorm2d: 3-24            [128, 128, 13, 13]        256\n",
       "│    │    └─ReLU: 3-25                   [128, 128, 13, 13]        --\n",
       "├─Sequential: 1-7                        [128, 256, 7, 7]          --\n",
       "│    └─BasicBlock_V0: 2-5                [128, 256, 7, 7]          --\n",
       "│    │    └─Conv2d: 3-26                 [128, 256, 7, 7]          294,912\n",
       "│    │    └─BatchNorm2d: 3-27            [128, 256, 7, 7]          512\n",
       "│    │    └─ReLU: 3-28                   [128, 256, 7, 7]          --\n",
       "│    │    └─Conv2d: 3-29                 [128, 256, 7, 7]          589,824\n",
       "│    │    └─BatchNorm2d: 3-30            [128, 256, 7, 7]          512\n",
       "│    │    └─Sequential: 3-31             [128, 256, 7, 7]          33,280\n",
       "│    │    └─ReLU: 3-32                   [128, 256, 7, 7]          --\n",
       "│    └─BasicBlock_V0: 2-6                [128, 256, 7, 7]          --\n",
       "│    │    └─Conv2d: 3-33                 [128, 256, 7, 7]          589,824\n",
       "│    │    └─BatchNorm2d: 3-34            [128, 256, 7, 7]          512\n",
       "│    │    └─ReLU: 3-35                   [128, 256, 7, 7]          --\n",
       "│    │    └─Conv2d: 3-36                 [128, 256, 7, 7]          589,824\n",
       "│    │    └─BatchNorm2d: 3-37            [128, 256, 7, 7]          512\n",
       "│    │    └─ReLU: 3-38                   [128, 256, 7, 7]          --\n",
       "├─Sequential: 1-8                        [128, 512, 4, 4]          --\n",
       "│    └─BasicBlock_V0: 2-7                [128, 512, 4, 4]          --\n",
       "│    │    └─Conv2d: 3-39                 [128, 512, 4, 4]          1,179,648\n",
       "│    │    └─BatchNorm2d: 3-40            [128, 512, 4, 4]          1,024\n",
       "│    │    └─ReLU: 3-41                   [128, 512, 4, 4]          --\n",
       "│    │    └─Conv2d: 3-42                 [128, 512, 4, 4]          2,359,296\n",
       "│    │    └─BatchNorm2d: 3-43            [128, 512, 4, 4]          1,024\n",
       "│    │    └─Sequential: 3-44             [128, 512, 4, 4]          132,096\n",
       "│    │    └─ReLU: 3-45                   [128, 512, 4, 4]          --\n",
       "│    └─BasicBlock_V0: 2-8                [128, 512, 4, 4]          --\n",
       "│    │    └─Conv2d: 3-46                 [128, 512, 4, 4]          2,359,296\n",
       "│    │    └─BatchNorm2d: 3-47            [128, 512, 4, 4]          1,024\n",
       "│    │    └─ReLU: 3-48                   [128, 512, 4, 4]          --\n",
       "│    │    └─Conv2d: 3-49                 [128, 512, 4, 4]          2,359,296\n",
       "│    │    └─BatchNorm2d: 3-50            [128, 512, 4, 4]          1,024\n",
       "│    │    └─ReLU: 3-51                   [128, 512, 4, 4]          --\n",
       "├─AdaptiveAvgPool2d: 1-9                 [128, 512, 1, 1]          --\n",
       "├─Flatten: 1-10                          [128, 512]                --\n",
       "├─Linear: 1-11                           [128, 9]                  4,617\n",
       "==========================================================================================\n",
       "Total params: 11,174,857\n",
       "Trainable params: 11,174,857\n",
       "Non-trainable params: 0\n",
       "Total mult-adds (Units.GIGABYTES): 54.48\n",
       "==========================================================================================\n",
       "Input size (MB): 5.12\n",
       "Forward/backward pass size (MB): 1089.22\n",
       "Params size (MB): 44.70\n",
       "Estimated Total Size (MB): 1139.04\n",
       "=========================================================================================="
      ]
     },
     "execution_count": 130,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from torchinfo import summary\n",
    "\n",
    "# Pass your exact instance name and expected tensor input dimensions\n",
    "summary(resnet_model_0, input_size=(128, 1, 100, 100))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "id": "2cb9c077-d0de-4809-93f6-43e97e8f71b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "def accuracy_fn(pred, truth):\n",
    "    total = len(pred)\n",
    "    correct = torch.eq(pred, truth).sum().item() \n",
    "    val = correct/total * 100\n",
    "    return val\n",
    "\n",
    "def train_step(model, \n",
    "              data_loader,\n",
    "              loss_fn,\n",
    "              opt,\n",
    "              device, epoch):\n",
    "    train_loss=0\n",
    "    train_acc=0\n",
    "\n",
    "    progress_bar = tqdm(data_loader, desc=f\"Epoch {epoch} Training\")\n",
    "    model.train()\n",
    "    for (X, y) in progress_bar:\n",
    "        X, y = X.to(device), y.to(device)\n",
    "\n",
    "        y_pred = model(X)\n",
    "        loss = loss_fn(y_pred, y)\n",
    "        train_loss+=loss.item()\n",
    "        train_acc += accuracy_fn(y_pred.argmax(dim=1), y)\n",
    "        opt.zero_grad()\n",
    "        loss.backward()\n",
    "        opt.step()\n",
    "        # scheduler.step()\n",
    "\n",
    "    train_loss = train_loss/len(data_loader)\n",
    "    train_acc /= len(data_loader)\n",
    "\n",
    "    print(f\"Epoch  {epoch} :   Train Loss = {train_loss} ----- Train Accuracy = {train_acc}%\")\n",
    "\n",
    "\n",
    "def test_step(model,\n",
    "             data_loader,\n",
    "             loss_fn,\n",
    "             device, epoch):\n",
    "\n",
    "    test_loss = 0\n",
    "    test_acc=0\n",
    "    model.eval()\n",
    "    \n",
    "    with torch.inference_mode():\n",
    "        for (X, y) in data_loader:\n",
    "            X, y = X.to(device), y.to(device)\n",
    "\n",
    "            y_pred = model(X)\n",
    "            loss=loss_fn(y_pred, y)\n",
    "            test_acc += accuracy_fn(y_pred.argmax(dim=1), y)\n",
    "            test_loss+=loss.item()\n",
    "            \n",
    "    test_loss = test_loss/len(data_loader)\n",
    "    test_acc /= len(data_loader)\n",
    "    print(f\"Epoch  {epoch} :   Test Loss = {test_loss} ----- Test Accuracy = {test_acc}%\")\n",
    "\n",
    "    return test_loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "id": "f20132c7-0d74-4c17-bd59-ba7078941921",
   "metadata": {},
   "outputs": [],
   "source": [
    "weights_tensor = torch.tensor(LABEL_WEIGHTS, dtype=torch.float32).to(device)\n",
    "\n",
    "loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)\n",
    "optimizer = torch.optim.AdamW(resnet_model_0.parameters(), lr=1e-3, weight_decay=1e-4)\n",
    "scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "id": "3c123071-3a9f-41c2-9150-60c1ab3dc724",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 1 Training: 100%|██████████████████████████████████████████████████████████████| 278/278 [00:34<00:00,  8.13it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  1 :   Train Loss = 0.8884454964519405 ----- Train Accuracy = 69.0945903991093%\n",
      "Epoch  1 :   Test Loss = 0.5156091074291751 ----- Test Accuracy = 81.55850319744205%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 2 Training: 100%|██████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.63it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  2 :   Train Loss = 0.5272002685413086 ----- Train Accuracy = 82.32525586673519%\n",
      "Epoch  2 :   Test Loss = 0.47326218937155157 ----- Test Accuracy = 80.49350698013018%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 3 Training: 100%|██████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.61it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  3 :   Train Loss = 0.42857166728098617 ----- Train Accuracy = 85.32995032545392%\n",
      "Epoch  3 :   Test Loss = 0.5054424813539862 ----- Test Accuracy = 81.79438591983556%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 4 Training: 100%|██████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.62it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  4 :   Train Loss = 0.3867199938717506 ----- Train Accuracy = 86.76027927657874%\n",
      "Epoch  4 :   Test Loss = 0.48379044229392526 ----- Test Accuracy = 88.35940176430285%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 5 Training: 100%|██████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.63it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  5 :   Train Loss = 0.36506720401829096 ----- Train Accuracy = 87.62320500742263%\n",
      "Epoch  5 :   Test Loss = 0.38170136050354664 ----- Test Accuracy = 84.20022910243235%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 6 Training: 100%|██████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.63it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  6 :   Train Loss = 0.31575305931430925 ----- Train Accuracy = 89.14899330535572%\n",
      "Epoch  6 :   Test Loss = 0.30950342357265864 ----- Test Accuracy = 90.31252676430285%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 7 Training: 100%|██████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.67it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  7 :   Train Loss = 0.28112862813923 ----- Train Accuracy = 89.87965877298161%\n",
      "Epoch  7 :   Test Loss = 0.37902437053996024 ----- Test Accuracy = 83.49195643485213%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 8 Training: 100%|██████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.69it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  8 :   Train Loss = 0.2743277184986811 ----- Train Accuracy = 90.30953808381865%\n",
      "Epoch  8 :   Test Loss = 0.3643723730560687 ----- Test Accuracy = 86.47898645369419%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 9 Training: 100%|██████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.68it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  9 :   Train Loss = 0.2716571624062473 ----- Train Accuracy = 90.78674560351719%\n",
      "Epoch  9 :   Test Loss = 0.2962517776667214 ----- Test Accuracy = 89.83215213543451%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 10 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.61it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  10 :   Train Loss = 0.25865454320534526 ----- Train Accuracy = 90.87176687221651%\n",
      "Epoch  10 :   Test Loss = 0.24797218579718536 ----- Test Accuracy = 88.95780696871074%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 11 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.63it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  11 :   Train Loss = 0.2516541275457084 ----- Train Accuracy = 91.02053178885463%\n",
      "Epoch  11 :   Test Loss = 0.47478803901149214 ----- Test Accuracy = 89.30101454550646%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 12 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.68it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  12 :   Train Loss = 0.2403675422956832 ----- Train Accuracy = 91.26265951524495%\n",
      "Epoch  12 :   Test Loss = 0.24947057927040744 ----- Test Accuracy = 89.2166177772068%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 13 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:33<00:00,  8.40it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  13 :   Train Loss = 0.2307278783790928 ----- Train Accuracy = 91.70324454721937%\n",
      "Epoch  13 :   Test Loss = 0.2168151440112282 ----- Test Accuracy = 91.2201488808953%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 14 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.68it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  14 :   Train Loss = 0.2261234855667936 ----- Train Accuracy = 91.92578972536258%\n",
      "Epoch  14 :   Test Loss = 0.23161686680835786 ----- Test Accuracy = 90.35749079307982%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 15 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.68it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  15 :   Train Loss = 0.22306957977602807 ----- Train Accuracy = 91.90076510220396%\n",
      "Epoch  15 :   Test Loss = 0.2268849440294204 ----- Test Accuracy = 92.9291388317917%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 16 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.68it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  16 :   Train Loss = 0.2232429893995575 ----- Train Accuracy = 91.82997352118306%\n",
      "Epoch  16 :   Test Loss = 0.22952351952306657 ----- Test Accuracy = 91.93993019869818%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 17 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.67it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  17 :   Train Loss = 0.21792863774374663 ----- Train Accuracy = 91.95670249514674%\n",
      "Epoch  17 :   Test Loss = 0.9855600836144077 ----- Test Accuracy = 85.09643178314491%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 18 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.67it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  18 :   Train Loss = 0.21973796116469577 ----- Train Accuracy = 92.16756059438164%\n",
      "Epoch  18 :   Test Loss = 0.2590931795751877 ----- Test Accuracy = 92.44858577423776%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 19 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.64it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  19 :   Train Loss = 0.21551542045722763 ----- Train Accuracy = 92.31958183453237%\n",
      "Epoch  19 :   Test Loss = 0.1961233692525102 ----- Test Accuracy = 93.24968596551331%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 20 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.62it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  20 :   Train Loss = 0.2061144505366147 ----- Train Accuracy = 92.43707712401508%\n",
      "Epoch  20 :   Test Loss = 0.21639330638237567 ----- Test Accuracy = 91.92297947356401%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 21 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.68it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  21 :   Train Loss = 0.19659634918081675 ----- Train Accuracy = 92.60859169807011%\n",
      "Epoch  21 :   Test Loss = 0.22143364826337897 ----- Test Accuracy = 92.18161185337445%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 22 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.69it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  22 :   Train Loss = 0.20341035971592228 ----- Train Accuracy = 92.46834675117049%\n",
      "Epoch  22 :   Test Loss = 0.17721921452086606 ----- Test Accuracy = 93.12032516843668%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 23 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.69it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  23 :   Train Loss = 0.1951543809243029 ----- Train Accuracy = 92.64231471965284%\n",
      "Epoch  23 :   Test Loss = 0.2322538679318248 ----- Test Accuracy = 90.64150465342013%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 24 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.68it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  24 :   Train Loss = 0.20803914181620098 ----- Train Accuracy = 92.54957641030033%\n",
      "Epoch  24 :   Test Loss = 0.17644134741410505 ----- Test Accuracy = 93.09494368790682%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 25 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:31<00:00,  8.69it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  25 :   Train Loss = 0.19143259257292575 ----- Train Accuracy = 92.7827380952381%\n",
      "Epoch  25 :   Test Loss = 0.20286715909731473 ----- Test Accuracy = 89.30627819173233%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 26 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.68it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  26 :   Train Loss = 0.18522477621654812 ----- Train Accuracy = 92.82743448098664%\n",
      "Epoch  26 :   Test Loss = 0.22986265765034036 ----- Test Accuracy = 90.11335574397624%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 27 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:31<00:00,  8.69it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  27 :   Train Loss = 0.18894406642577202 ----- Train Accuracy = 93.07545035400251%\n",
      "Epoch  27 :   Test Loss = 0.19240029397413885 ----- Test Accuracy = 92.59471886776294%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 28 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:31<00:00,  8.69it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  28 :   Train Loss = 0.18320619359130053 ----- Train Accuracy = 92.9010363138061%\n",
      "Epoch  28 :   Test Loss = 0.3339039209774501 ----- Test Accuracy = 91.14743919150393%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 29 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:31<00:00,  8.69it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  29 :   Train Loss = 0.18418313541429507 ----- Train Accuracy = 92.99069672833163%\n",
      "Epoch  29 :   Test Loss = 0.18965731223495744 ----- Test Accuracy = 91.98217319001941%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 30 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:31<00:00,  8.69it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  30 :   Train Loss = 0.17542863015433868 ----- Train Accuracy = 93.22113737581364%\n",
      "Epoch  30 :   Test Loss = 0.20053949863683407 ----- Test Accuracy = 91.04310301758593%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 31 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.68it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  31 :   Train Loss = 0.18215228756554694 ----- Train Accuracy = 93.21007479730501%\n",
      "Epoch  31 :   Test Loss = 0.17332960142613316 ----- Test Accuracy = 93.05551094838415%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 32 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:31<00:00,  8.69it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  32 :   Train Loss = 0.1846246546228155 ----- Train Accuracy = 93.1002073341327%\n",
      "Epoch  32 :   Test Loss = 0.18071560981087118 ----- Test Accuracy = 93.7694041195615%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 33 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.65it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  33 :   Train Loss = 0.17157896980643272 ----- Train Accuracy = 93.40960267500286%\n",
      "Epoch  33 :   Test Loss = 0.21386639607681646 ----- Test Accuracy = 89.15760248943702%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 34 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.54it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  34 :   Train Loss = 0.1719298155160902 ----- Train Accuracy = 93.17046362909673%\n",
      "Epoch  34 :   Test Loss = 0.19493911219693774 ----- Test Accuracy = 92.94881059438164%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 35 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.66it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  35 :   Train Loss = 0.16935107412098124 ----- Train Accuracy = 93.5359747915953%\n",
      "Epoch  35 :   Test Loss = 0.24221237317370853 ----- Test Accuracy = 92.24633685908415%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 36 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.68it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  36 :   Train Loss = 0.1720088148060784 ----- Train Accuracy = 93.34496688363595%\n",
      "Epoch  36 :   Test Loss = 0.18573604873884067 ----- Test Accuracy = 92.2575778662784%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 37 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.68it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  37 :   Train Loss = 0.16901451471201379 ----- Train Accuracy = 93.56688756137946%\n",
      "Epoch  37 :   Test Loss = 0.17252018612226566 ----- Test Accuracy = 94.528172105173%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 38 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.68it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  38 :   Train Loss = 0.16569159401031278 ----- Train Accuracy = 93.61738287941075%\n",
      "Epoch  38 :   Test Loss = 0.33761496665023216 ----- Test Accuracy = 83.4435576538769%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 39 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.68it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  39 :   Train Loss = 0.17005604285773612 ----- Train Accuracy = 93.45465591812264%\n",
      "Epoch  39 :   Test Loss = 0.1560351429708141 ----- Test Accuracy = 93.62028234555213%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 40 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.68it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  40 :   Train Loss = 0.16162759357755133 ----- Train Accuracy = 93.60078901164782%\n",
      "Epoch  40 :   Test Loss = 0.16047414727241016 ----- Test Accuracy = 93.9857935080507%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 41 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.68it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  41 :   Train Loss = 0.1562696685143512 ----- Train Accuracy = 93.7103888317917%\n",
      "Epoch  41 :   Test Loss = 0.19375280639250503 ----- Test Accuracy = 93.49109997716113%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 42 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.68it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  42 :   Train Loss = 0.16152800689283892 ----- Train Accuracy = 93.74139081591869%\n",
      "Epoch  42 :   Test Loss = 0.37585210845732003 ----- Test Accuracy = 90.14127983327624%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 43 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.66it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  43 :   Train Loss = 0.16360222349921577 ----- Train Accuracy = 93.80839078737012%\n",
      "Epoch  43 :   Test Loss = 0.18281283878165183 ----- Test Accuracy = 93.80022767500286%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 44 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.67it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  44 :   Train Loss = 0.16136807724565483 ----- Train Accuracy = 93.81989943759278%\n",
      "Epoch  44 :   Test Loss = 0.14445853592466107 ----- Test Accuracy = 93.48249079307982%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 45 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.68it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  45 :   Train Loss = 0.1612172016887356 ----- Train Accuracy = 93.79723899451868%\n",
      "Epoch  45 :   Test Loss = 0.1422268336759411 ----- Test Accuracy = 94.60966940733128%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 46 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.67it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  46 :   Train Loss = 0.14834764979029302 ----- Train Accuracy = 93.95180284343954%\n",
      "Epoch  46 :   Test Loss = 0.13726317302303778 ----- Test Accuracy = 94.08415232100035%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 47 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.68it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  47 :   Train Loss = 0.14990373479751803 ----- Train Accuracy = 94.05042929941762%\n",
      "Epoch  47 :   Test Loss = 0.1791411863519348 ----- Test Accuracy = 93.01634585188991%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 48 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.68it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  48 :   Train Loss = 0.149495364240498 ----- Train Accuracy = 93.99422426344638%\n",
      "Epoch  48 :   Test Loss = 0.1629359140965364 ----- Test Accuracy = 93.80303792680142%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 49 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:31<00:00,  8.69it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  49 :   Train Loss = 0.14865041084641176 ----- Train Accuracy = 94.08134206920178%\n",
      "Epoch  49 :   Test Loss = 0.18133843599034727 ----- Test Accuracy = 93.54748344181797%\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Epoch 50 Training: 100%|█████████████████████████████████████████████████████████████| 278/278 [00:32<00:00,  8.68it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch  50 :   Train Loss = 0.15482284332350862 ----- Train Accuracy = 93.9352089756766%\n",
      "Epoch  50 :   Test Loss = 0.14027068754591102 ----- Test Accuracy = 95.00310465912985%\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from tqdm import tqdm\n",
    "from sklearn.metrics import f1_score, confusion_matrix\n",
    "\n",
    "EPOCHS = 50\n",
    "MODEL_SAVE_PATH = r\"C:\\Users\\HP\\Desktop\\Wafer\\Models\\\\\"\n",
    "model_name = \"Resnet_Wafer_V0_\"\n",
    "MODEL_SAVE_PATH += model_name\n",
    "\n",
    "for epoch in range(1, EPOCHS+1):\n",
    "    train_step(resnet_model_0, train_loader, loss_fn, opt, device, epoch)\n",
    "\n",
    "    v_loss = test_step(resnet_model_0, val_loader, loss_fn, device, epoch)\n",
    "    print()\n",
    "    \n",
    "    scheduler.step(v_loss)\n",
    "    torch.cuda.empty_cache()\n",
    "    if((epoch+1)%3==0):\n",
    "        model_path_curr = MODEL_SAVE_PATH +\"epoch_\" + str(epoch) + \".pth\"\n",
    "        torch.save(obj=resnet_model_0.state_dict(), f=model_path_curr)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "id": "6964037c-f881-4a57-abbf-3b65fc44741b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<All keys matched successfully>"
      ]
     },
     "execution_count": 148,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "load_model = ResNet_Wafer_V0()\n",
    "load_model.load_state_dict(torch.load(f=r\"C:\\Users\\HP\\Desktop\\Wafer\\Models\\Resnet_Wafer_V0_epoch_50.pth\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "88ea35af-99fc-4063-bd60-2fd187a02591",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
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
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
