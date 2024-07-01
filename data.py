import concurrent.futures
import io

import pandas as pd
from datasets import (get_dataset_config_names,
                      load_dataset)
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class BaseDataset(Dataset):
    def __init__(self, split):
        self._split = split
        self.data = []
        self.task_prompt = ""

    def __len__(self):
        return len(self.data)

    def correct_casing_finqa(self, text, is_question=False):
        if text and text[0].islower():
            text = text.capitalize()
        if not text.endswith(".") and not is_question:
            text += "."
        if not text.endswith("?") and is_question:
            text += "?"
        return text


class DocVQADataset(BaseDataset):
    def __init__(self, split):
        super().__init__(split)
        self.data = load_dataset("HuggingFaceM4/DocumentVQA", split=split)
        self.task_prompt = "<DocVQA>"

    def __getitem__(self, idx):
        example = self.data[idx]
        question = self.task_prompt + self.correct_casing_finqa(
            example["question"], True
        )
        first_answer = example["answers"][0]
        answers = self.correct_casing_finqa(first_answer)
        image = example["image"]  # The image is already a PIL Image object
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, answers, image
    

class TheCauldronDataset(BaseDataset):
    def __init__(self, split):
        super().__init__(split)
        self.images_df, self.texts_df = self.load_all_configs(split)
        self.task_prompt = "<VQA>"

    def __len__(self):
        return len(self.texts_df)
    
    def load_config(self, config_name, split):
        print(f"Loading config: {config_name}")
        dataset = load_dataset("HuggingFaceM4/the_cauldron", config_name, split=split)
        print(f"Finished loading config: {config_name}")

        df_data = dataset.to_pandas()

        # Create the images DataFrame
        df_images = df_data[['images']].copy()
        df_images['image_index'] = df_images.index

        # Explode the texts into separate rows and create a DataFrame
        df_texts = df_data[['texts']].explode('texts').reset_index()
        df_texts.rename(columns={'index': 'image_index'}, inplace=True)

        # Extract 'user', 'assistant', and 'source' from the 'texts' column
        df_texts['question'] = df_texts['texts'].apply(lambda x: x.get('user'))
        df_texts['answer'] = df_texts['texts'].apply(lambda x: x.get('assistant'))
        df_texts['source'] = df_texts['texts'].apply(lambda x: x.get('source'))

        # Drop the original 'texts' column
        df_texts.drop(columns=['texts'], inplace=True)

        # Copy the 'source' column to the images df, using the first source per image index
        df_images = df_images.merge(df_texts[['image_index', 'source']], on='image_index', how='left')
        print(f"Finished processing config: {config_name}")

        return df_images, df_texts

    def load_all_configs(self, split):
        cauldron_config_names = get_dataset_config_names("HuggingFaceM4/the_cauldron")

        images_dfs = []
        texts_dfs = []

        # Use ThreadPoolExecutor for parallel processing and tqdm for progress tracking
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:  # Limit the number of workers
            with tqdm(total=len(cauldron_config_names), desc="Total Progress") as total_pbar:
                futures = {executor.submit(self.load_config, config_name, split): config_name for config_name in cauldron_config_names}
                for future in concurrent.futures.as_completed(futures):
                    config_name = futures[future]
                    try:
                        df_images, df_texts = future.result()
                        images_dfs.append(df_images)
                        texts_dfs.append(df_texts)
                    except Exception as exc:
                        print(f"{config_name} generated an exception: {exc}")
                    total_pbar.update(1)

        # Merge all the loaded DataFrames
        print("Merging DataFrames...")
        merged_images_df = pd.concat(images_dfs, ignore_index=True)
        merged_texts_df = pd.concat(texts_dfs, ignore_index=True)
        print("Finished merging DataFrames")

        return merged_images_df, merged_texts_df

    def __getitem__(self, idx):
        example = self.texts_df.iloc[idx]
        question = example["question"]
        answer = example["answer"]
        source = example["source"]
        image_idx = example["image_index"]

        image_data = self.images_df.loc[(self.images_df['image_index'] == image_idx) & (self.images_df['source'] == source), 'images'].values[0][0]['bytes'] 
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != "RGB":
            image = image.convert("RGB")

        return question, answer, image



class ObjectDetectionDataset(BaseDataset):
    def __init__(self, split, processor, name="rishitdagli/cppe-5", class_list=["coverall", "face shield", "glove", "goggle", "mask"]):
        self.task_prompt = "<OD>"
        #self.img_paths = []
        self.labels = []
        with open('/data/pic/det/lig/tttrain.txt', 'r') as fd:
            self.img_paths = fd.readlines()
        random.shuffle(self.img_paths)

    def read_annotations(self,annotation_path,width,height):
        bboxes = []
        categories = []
        with open(annotation_path, 'r') as file:
           for line in file:
              parts = line.strip().split()
              category = int(parts[0])
              bbox = list(map(float, parts[1:]))
            # Denormalize bbox
              x_center = bbox[0] * width
              y_center = bbox[1] * height
              bbox_width = bbox[2] * width
              bbox_height = bbox[3] * height
              xmin = x_center - bbox_width / 2
              ymin = y_center - bbox_height / 2
              xmax = x_center + bbox_width / 2
              ymax = y_center + bbox_height / 2
              denormalized_bbox = [xmin, ymin, xmax, ymax]
              categories.append(category)
              bboxes.append(denormalized_bbox)
        return bboxes, categories


    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_path = img_path.replace('\n', '')
        task = self.task_prompt

        #if img_path.lower().endswith('.jpg'):
    # 替换扩展名为'.txt'
        img_txt = img_path[:-4] + '.txt'
        image=Image.open(img_path)
        width, height = image.size
        #print(img_txt)   

        bboxes,categories= self.read_annotations(img_txt,width,height)
 width, height = image.size
        bins_w, bins_h = [1000, 1000]  # Quantization bins.
        size_per_bin_w = width / bins_w
        size_per_bin_h = height / bins_h


        bbox_str = ""
        for (cat, bbox) in zip(categories, bboxes):
            # if len(bbox_str) == 0:
            bbox_str +="badlig" #self.class_list[cat]
            #bbox_str += self.class_list[cat]
            bbox = bbox.copy()

            xmin, ymin, xmax, ymax = torch.tensor(bbox).split(1, dim=-1)

            quantized_xmin = (
                xmin / size_per_bin_w).floor().clamp(0, bins_w - 1)
            quantized_ymin = (
                ymin / size_per_bin_h).floor().clamp(0, bins_h - 1)
            quantized_xmax = (
                xmax / size_per_bin_w).floor().clamp(0, bins_w - 1)
            quantized_ymax = (
                ymax / size_per_bin_h).floor().clamp(0, bins_h - 1)

            quantized_boxes = torch.cat(
                (quantized_xmin, quantized_ymin, quantized_xmax, quantized_ymax), dim=-1
            ).int()

            bbox_str += str(f"<loc_{quantized_boxes[0]}><loc_{quantized_boxes[1]}><loc_{quantized_boxes[2]}><loc_{quantized_boxes[3]}>")

            # bbox_formatted_list.append(bbox_str)
         #   print(bbox_str)

        if image.mode != "RGB":
            image = image.convert("RGB")
        return task, bbox_str, image

