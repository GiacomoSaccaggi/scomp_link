
from scomp_link.utils.logger import get_logger
logger = get_logger(__name__)

# -*- coding: utf-8 -*-
"""

 ██████╗██╗   ██╗██████╗ ███████╗██████╗ ██╗   ██╗██╗ ██████╗███████╗██████╗ 
██╔════╝██║   ██║██╔══██╗██╔════╝██╔══██╗██║   ██║██║██╔════╝██╔════╝██╔══██╗
╚█████╗ ██║   ██║██████╔╝█████╗  ██████╔╝╚██╗ ██╔╝██║╚█████╗ █████╗  ██║  ██║
 ╚═══██╗██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗ ╚████╔╝ ██║ ╚═══██╗██╔══╝  ██║  ██║
██████╔╝╚██████╔╝██║     ███████╗██║  ██║  ╚██╔╝  ██║██████╔╝███████╗██████╔╝
╚═════╝  ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝╚═════╝ ╚══════╝╚═════╝ 

████████╗███████╗██╗  ██╗████████╗
╚══██╔══╝██╔════╝╚██╗██╔╝╚══██╔══╝
   ██║   █████╗   ╚███╔╝    ██║   
   ██║   ██╔══╝   ██╔██╗    ██║   
   ██║   ███████╗██╔╝╚██╗   ██║   
   ╚═╝   ╚══════╝╚═╝  ╚═╝   ╚═╝   
"""


class SpacyEmbeddingModel(object):
    """
    Text classification using Spacy and Contrastive Embeddings
    """
    def __init__(self, lan: str = 'en', model_name: str = 'bert-base-uncased'):
        import os
        import sys
        import torch
        from transformers import AutoTokenizer, AutoModel
        from .contrastive_net import ContrastiveSiameseModel

        self.all_spacy_lan = ['zh', 'da', 'nl', 'en', 'fr', 'de', 'el', 'it', 'ja', 'lt', 'nb', 'pl', 'pt', 'ro', 'ru',
                              'af', 'sq', 'ar', 'hy', 'eu', 'bn', 'bg', 'ca', 'hr', 'cs', 'et', 'fi', 'gu', 'he', 'hi',
                              'id', 'ga', 'kn', 'ko', 'lv', 'lij', 'lb', 'mk', 'ml', 'mr', 'ne', 'fa', 'sa', 'sr', 'si',
                              'sl', 'sk', 'es', 'sv', 'hu', 'tl', 'ta', 'tt', 'te', 'th', 'tr', 'uk', 'ur', 'vi', 'yo']
        self.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

        result = self._select_language(lan, 'md')
        if not result or len(result) != 2:
            raise ValueError(f"Failed to load spaCy model for language '{lan}'. "
                             f"Try: {sys.executable} -m spacy download {lan}_core_web_md")
        self.nlp, self.stop_words = result
        
        # Initialize Contrastive Model for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)
        self.contrastive_model = ContrastiveSiameseModel(self.base_model)

    def _select_language(self, lan: 'Language in ISO 639-1', dict_size: "['sm', 'md', 'lg', 'trf']" = 'sm'):
        """
        Internal function for selecting languages
        :param lan: Language in ISO 639-1
        :param dict_size: size of dictionary
        :return: spacy dict and stopwords
        """
        import sys
        import spacy
        import subprocess
        if lan in self.all_spacy_lan:
            nlp = 'not find'
            STOP_WORDS = ''
            try:
                nlp = spacy.load(f'{lan}_core_web_{dict_size}')
            except Exception as e:
                if nlp != 'not find':
                    logger.info(e)
                pass
            if nlp == 'not find':
                try:
                    nlp = spacy.load(f'{lan}_core_news_{dict_size}')
                except Exception as e:
                    if nlp != 'not find':
                        logger.info(e)
                    pass
            if nlp == 'not find':
                try:
                    subprocess.call(f'{sys.executable} -m spacy download {lan}_core_news_{dict_size}', shell=True)
                    nlp = spacy.load(f'{lan}_core_news_{dict_size}')
                except Exception as e:
                    if nlp != 'not find':
                        logger.info(e)
                    pass
            if nlp == 'not find':
                try:
                    subprocess.call(f'{sys.executable} -m spacy download {lan}_core_web_{dict_size}', shell=True)
                    nlp = spacy.load(f'{lan}_core_web_{dict_size}')
                except Exception as e:
                    if nlp != 'not find':
                        logger.info(e)
                    logger.info('\033[91m\tError:\n\t-\tFailed download of dictionary\n\t\t-\tThe dictionary size does'
                          ' not exist\n\t\t-\tSpacy does not support this language\n\n\t\tView all languages in '
                          'website:\n\t\thttps://spacy.io/usage/models\033[0m\033[0m')
                return []
            # Dynamic stop words import for any supported spaCy language
            import importlib
            try:
                lang_module = importlib.import_module(f'spacy.lang.{lan}.stop_words')
                STOP_WORDS = lang_module.STOP_WORDS
            except (ImportError, ModuleNotFoundError):
                STOP_WORDS = set()

            return nlp, STOP_WORDS
        else:
            logger.info(f'\033[91m\tLanguage not managed, select in this languages:\n\t{",".join(self.all_spacy_lan)}\n\t'
                  'View all languages in website:\n\t\thttps://spacy.io/usage/models\033[0m')
            return []

    def __extract_words(self, sentence: str):
        """
        Extract verbs, nouns and proper nouns from sentence
        """
        import re
        sentence = re.sub('[^A-Za-z]+', ' ', sentence.replace("\n", " ")).lower()
        mytokens = self.nlp(sentence)
        words = [word.lemma_.lower().strip() for word in mytokens if (word.pos_ == "VERB" or word.pos_ == "AUX"
                 or word.pos_ == "NOUN" or word.pos_ == "PROPN") and word not in self.stop_words]
        if len(words) == 0:
            return [word.lemma_.lower().strip() for word in mytokens]
        else:
            return words

    def report_progress(self, epoch, best, losses, scores):
        """
        Report training progress
        """
        logger.info(
            "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(
                losses["textcat"],
                scores["textcat_p"],
                scores["textcat_r"],
                scores["textcat_f"],
            )
        )

    def evaluate_textcat(self, tokenizer, textcat, texts, cats):
        """
        Evaluate text categorization model
        """
        import numpy as np
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
        previsioni = []
        lab = []
        docs = (tokenizer(text) for text in texts)
        for i, doc in enumerate(textcat.pipe(docs)):
            previsioni.append(np.argmax([score for label, score in doc.cats.items()]))
            lab.append([label for label, score in doc.cats.items()])
        real = []
        for row in cats:
            real.append(np.where([i for prev, i in row.items()])[0][0])

        precision = precision_score(real, previsioni, average='weighted')
        recall = recall_score(y_true=real, y_pred=previsioni, average='weighted')
        f_score = f1_score(y_true=real, y_pred=previsioni, average='weighted')
        acc = accuracy_score(y_true=real, y_pred=previsioni)
        return {
            "textcat_p": precision,
            "textcat_r": recall,
            "textcat_f": f_score,
            "acc": acc,
        }

    def get_opt_params(self, kwargs):
        """
        Get optimizer parameters
        """
        return {
            "learn_rate": kwargs["learn_rate"],
            "optimizer_B1": kwargs["b1"],
            "optimizer_B2": kwargs["b1"] * kwargs["b2_ratio"],
            "optimizer_eps": kwargs["adam_eps"],
            "L2": kwargs["L2"],
            "grad_norm_clip": kwargs["grad_norm_clip"],
        }

    def configure_optimizer(self, opt, params):
        """
        Configure optimizer with parameters
        """
        opt.alpha = params["learn_rate"]
        opt.b1 = params["optimizer_B1"]
        opt.b2 = params["optimizer_B2"]
        opt.eps = params["optimizer_B2"]
        opt.L2 = params["L2"]
        opt.max_grad_norm = params["grad_norm_clip"]

    def training(self, to_tag, tagged, categorie):
        """
        Training wrapper
        """
        from sklearn.model_selection import train_test_split
        texts = []
        y = []
        for k, v in to_tag.items():
            texts.append(v)
            y.append(tagged[k])
        x_train, x_test, y_train, y_test = train_test_split(texts, y, test_size=0.33, random_state=42)

        return self.cnn_embedding_textcategorizer(x_train, y_train, x_test, y_test, categorie)

    def cnn_embedding_textcategorizer(self, texts, y_lab, texts_test, y_lab_test, classes, width=16, embed_size=75,
                                      patience=20, epoch=100, learn_rate=0.1, dropout=0.2, batch_size=8, b1=0.0,
                                      b2_ratio=0.0, adam_eps=0.0, L2=0.0, grad_norm_clip=1.0, use_tqdm=True):
        """
        Train CNN text categorizer with embeddings.
        Compatible with spaCy v3 API.
        """
        from spacy.util import minibatch
        from spacy.training import Example
        import numpy as np
        import tqdm
        categ = np.asarray(classes)
        nr_categ = len(categ)
        texts = list(texts)
        y_lab = list(y_lab)
        texts_test = list(texts_test)
        y_lab_test = list(y_lab_test)

        def __load_textcat_data(texts_, y_lab_, texts_test_, y_lab_test_):
            """Load and prepare text categorization data."""
            y_lab2 = [int(np.where(ny == categ)[0]) for ny in y_lab_]
            y_lab2_test = [int(np.where(ny == categ)[0]) for ny in y_lab_test_]
            train_texts_ = [' '.join(self.__extract_words(msg)) for msg in texts_]
            eval_texts_ = [' '.join(self.__extract_words(msg)) for msg in texts_test_]
            train_labels = y_lab2
            eval_labels = y_lab2_test

            def __gen_dict_label(y):
                d = dict()
                for ncat in range(0, nr_categ):
                    if y == ncat:
                        d[str(categ[ncat])] = True
                    else:
                        d[str(categ[ncat])] = False
                return d
            train_cats_ = [__gen_dict_label(y) for y in train_labels]
            eval_cats_ = [__gen_dict_label(y) for y in eval_labels]
            return (train_texts_, train_cats_), (eval_texts_, eval_cats_)

        # Create a separate blank model for textcat training
        # (avoids lookup reinit issues with loaded models in spaCy v3)
        import spacy
        textcat_nlp = spacy.blank(self.nlp.lang)
        textcat = textcat_nlp.add_pipe("textcat", last=True)

        for cl in categ:
            textcat.add_label(str(cl))

        (train_texts, train_cats), (dev_texts, dev_cats) = __load_textcat_data(texts, y_lab, texts_test, y_lab_test)
        logger.info(
            "Number of examples ({} training, {} evaluation)".format(
                len(train_texts), len(dev_texts)
            )
        )

        # Build Example objects for spaCy v3 training
        train_examples = []
        for text, cats in zip(train_texts, train_cats):
            doc = textcat_nlp.make_doc(text)
            train_examples.append(Example.from_dict(doc, {"cats": cats}))

        best_acc = 0.0

        class EarlyStopping(object):
            """Early stopping callback."""
            def __init__(self, metric, patience_):
                self.metric = metric
                self.max_patience = patience_
                self.current_patience = patience_
                self.best = 0.5

            def update(self, result):
                """Update early stopping state."""
                if result[self.metric] >= self.best:
                    self.best = result[self.metric]
                    self.current_patience = self.max_patience
                    return False
                else:
                    self.current_patience -= 1
                    return self.current_patience <= 0

        early_stopping = EarlyStopping("acc", patience)

        # Initialize the pipeline for training (spaCy v3)
        textcat_nlp.initialize(lambda: train_examples)

        logger.info("Training the model...")
        if True:  # training block
            logger.info("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
            for i in range(epoch):
                losses = {}
                # Shuffle training data each epoch
                np.random.shuffle(train_examples)

                if use_tqdm:
                    try:
                        batches_iter = tqdm.tqdm(
                            list(minibatch(train_examples, size=batch_size)), leave=False
                        )
                    except Exception:
                        batches_iter = list(minibatch(train_examples, size=batch_size))
                else:
                    batches_iter = list(minibatch(train_examples, size=batch_size))

                for batch in batches_iter:
                    textcat_nlp.update(batch, drop=dropout, losses=losses)

                # Evaluate on dev set
                scores = self.evaluate_textcat(textcat_nlp.tokenizer, textcat, dev_texts, dev_cats)
                scores1 = self.evaluate_textcat(textcat_nlp.tokenizer, textcat, train_texts, train_cats)
                logger.info('Epoch ' + str(i))
                logger.info('\nResults Test:\n')
                best_acc = max(best_acc, scores["acc"])
                self.report_progress(i, best_acc, losses, {"textcat": losses.get("textcat", 0),
                                                           "textcat_p": scores["textcat_p"],
                                                           "textcat_r": scores["textcat_r"],
                                                           "textcat_f": scores["textcat_f"]})
                logger.info('\nResults Train:\n')
                best_acc = max(best_acc, scores1["acc"])
                self.report_progress(i, best_acc, losses, {"textcat": losses.get("textcat", 0),
                                                           "textcat_p": scores1["textcat_p"],
                                                           "textcat_r": scores1["textcat_r"],
                                                           "textcat_f": scores1["textcat_f"]})
                should_stop = early_stopping.update(scores)
                if should_stop:
                    logger.info('The model does not learn, try to change CNN architecture')
                    break

        logger.info('Predicting values train...')
        ytrain = []
        for ytr in train_cats:
            ytrain.append([label for label, val in ytr.items() if val][0])
        ytest = []
        for yte in dev_cats:
            ytest.append([label for label, val in yte.items() if val][0])

        logger.info('Predicting values test...')
        fitted_y_test = []
        for val in dev_texts:
            doc = textcat_nlp(val)
            fitted_y_test.append(max(doc.cats, key=doc.cats.get))
        fitted_y_train = []
        for val in train_texts:
            doc = textcat_nlp(val)
            fitted_y_train.append(max(doc.cats, key=doc.cats.get))

        logger.info('Generating reports...')
        from sklearn.metrics import classification_report, confusion_matrix

        clasification_report_tr = classification_report(ytrain, fitted_y_train)
        confusion_matrix_tr = confusion_matrix(ytrain, fitted_y_train)
        clasification_report_te = classification_report(ytest, fitted_y_test)
        confusion_matrix_te = confusion_matrix(ytest, fitted_y_test)

        previsioni_scores = {'classification_report_tr': clasification_report_tr,
                             'confusion_matrix_tr': confusion_matrix_tr,
                             'classification_report_te': clasification_report_te,
                             'confusion_matrix_te': confusion_matrix_te}
        
        return textcat_nlp, previsioni_scores
