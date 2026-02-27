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
        import torch
        from transformers import AutoTokenizer, AutoModel
        from .contrastive_net import ContrastiveSiameseModel

        self.all_spacy_lan = ['zh', 'da', 'nl', 'en', 'fr', 'de', 'el', 'it', 'ja', 'lt', 'nb', 'pl', 'pt', 'ro', 'ru',
                              'af', 'sq', 'ar', 'hy', 'eu', 'bn', 'bg', 'ca', 'hr', 'cs', 'et', 'fi', 'gu', 'he', 'hi',
                              'id', 'ga', 'kn', 'ko', 'lv', 'lij', 'lb', 'mk', 'ml', 'mr', 'ne', 'fa', 'sa', 'sr', 'si',
                              'sl', 'sk', 'es', 'sv', 'hu', 'tl', 'ta', 'tt', 'te', 'th', 'tr', 'uk', 'ur', 'vi', 'yo']
        self.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

        self.nlp, self.stop_words = self._select_language(lan, 'md')
        
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
        import spacy
        import subprocess
        if lan in self.all_spacy_lan:
            nlp = 'not find'
            STOP_WORDS = ''
            try:
                nlp = spacy.load(f'{lan}_core_web_{dict_size}')
            except Exception as e:
                if nlp != 'not find':
                    print(e)
                pass
            if nlp == 'not find':
                try:
                    nlp = spacy.load(f'{lan}_core_news_{dict_size}')
                except Exception as e:
                    if nlp != 'not find':
                        print(e)
                    pass
            if nlp == 'not find':
                try:
                    subprocess.call(f'python -m spacy download {lan}_core_news_{dict_size}', shell=True)
                    nlp = spacy.load(f'{lan}_core_news_{dict_size}')
                except Exception as e:
                    if nlp != 'not find':
                        print(e)
                    pass
            if nlp == 'not find':
                try:
                    subprocess.call(f'python -m spacy download {lan}_core_web_{dict_size}', shell=True)
                    nlp = spacy.load(f'{lan}_core_web_{dict_size}')
                except Exception as e:
                    if nlp != 'not find':
                        print(e)
                    print('\033[91m\tError:\n\t-\tFailed download of dictionary\n\t\t-\tThe dictionary size does'
                          ' not exist\n\t\t-\tSpacy does not support this language\n\n\t\tView all languages in '
                          'website:\n\t\thttps://spacy.io/usage/models\033[0m\033[0m')
                return []
            if lan == 'zh':
                from spacy.lang.zh.stop_words import STOP_WORDS
            elif lan == 'da':
                from spacy.lang.da.stop_words import STOP_WORDS
            elif lan == 'nl':
                from spacy.lang.nl.stop_words import STOP_WORDS
            elif lan == 'en':
                from spacy.lang.en.stop_words import STOP_WORDS
            elif lan == 'fr':
                from spacy.lang.fr.stop_words import STOP_WORDS
            elif lan == 'de':
                from spacy.lang.de.stop_words import STOP_WORDS
            elif lan == 'el':
                from spacy.lang.el.stop_words import STOP_WORDS
            elif lan == 'it':
                from spacy.lang.it.stop_words import STOP_WORDS
            elif lan == 'ja':
                from spacy.lang.ja.stop_words import STOP_WORDS
            elif lan == 'lt':
                from spacy.lang.lt.stop_words import STOP_WORDS
            elif lan == 'nb':
                from spacy.lang.nb.stop_words import STOP_WORDS
            elif lan == 'pl':
                from spacy.lang.pl.stop_words import STOP_WORDS
            elif lan == 'pt':
                from spacy.lang.pt.stop_words import STOP_WORDS
            elif lan == 'ro':
                from spacy.lang.ro.stop_words import STOP_WORDS
            elif lan == 'ru':
                from spacy.lang.ru.stop_words import STOP_WORDS
            elif lan == 'es':
                from spacy.lang.es.stop_words import STOP_WORDS
            elif lan == 'af':
                from spacy.lang.af.stop_words import STOP_WORDS
            elif lan == 'sq':
                from spacy.lang.sq.stop_words import STOP_WORDS
            elif lan == 'ar':
                from spacy.lang.ar.stop_words import STOP_WORDS
            elif lan == 'hy':
                from spacy.lang.hy.stop_words import STOP_WORDS
            elif lan == 'eu':
                from spacy.lang.eu.stop_words import STOP_WORDS
            elif lan == 'bn':
                from spacy.lang.bn.stop_words import STOP_WORDS
            elif lan == 'bg':
                from spacy.lang.bg.stop_words import STOP_WORDS
            elif lan == 'ca':
                from spacy.lang.ca.stop_words import STOP_WORDS
            elif lan == 'hr':
                from spacy.lang.hr.stop_words import STOP_WORDS
            elif lan == 'cs':
                from spacy.lang.cs.stop_words import STOP_WORDS
            elif lan == 'et':
                from spacy.lang.et.stop_words import STOP_WORDS
            elif lan == 'fi':
                from spacy.lang.fi.stop_words import STOP_WORDS
            elif lan == 'gu':
                from spacy.lang.gu.stop_words import STOP_WORDS
            elif lan == 'he':
                from spacy.lang.he.stop_words import STOP_WORDS
            elif lan == 'hi':
                from spacy.lang.hi.stop_words import STOP_WORDS
            elif lan == 'hu':
                from spacy.lang.hu.stop_words import STOP_WORDS
            elif lan == 'id':
                from spacy.lang.id.stop_words import STOP_WORDS
            elif lan == 'ga':
                from spacy.lang.ga.stop_words import STOP_WORDS
            elif lan == 'kn':
                from spacy.lang.kn.stop_words import STOP_WORDS
            elif lan == 'ko':
                from spacy.lang.ko.stop_words import STOP_WORDS
            elif lan == 'lv':
                from spacy.lang.lv.stop_words import STOP_WORDS
            elif lan == 'lij':
                from spacy.lang.lij.stop_words import STOP_WORDS
            elif lan == 'lb':
                from spacy.lang.lb.stop_words import STOP_WORDS
            elif lan == 'mk':
                from spacy.lang.mk.stop_words import STOP_WORDS
            elif lan == 'ml':
                from spacy.lang.ml.stop_words import STOP_WORDS
            elif lan == 'mr':
                from spacy.lang.mr.stop_words import STOP_WORDS
            elif lan == 'ne':
                from spacy.lang.ne.stop_words import STOP_WORDS
            elif lan == 'fa':
                from spacy.lang.fa.stop_words import STOP_WORDS
            elif lan == 'sa':
                from spacy.lang.sa.stop_words import STOP_WORDS
            elif lan == 'sr':
                from spacy.lang.sr.stop_words import STOP_WORDS
            elif lan == 'si':
                from spacy.lang.si.stop_words import STOP_WORDS
            elif lan == 'sk':
                from spacy.lang.sk.stop_words import STOP_WORDS
            elif lan == 'sl':
                from spacy.lang.sl.stop_words import STOP_WORDS
            elif lan == 'sv':
                from spacy.lang.sv.stop_words import STOP_WORDS
            elif lan == 'tl':
                from spacy.lang.tl.stop_words import STOP_WORDS
            elif lan == 'ta':
                from spacy.lang.ta.stop_words import STOP_WORDS
            elif lan == 'tt':
                from spacy.lang.tt.stop_words import STOP_WORDS
            elif lan == 'te':
                from spacy.lang.te.stop_words import STOP_WORDS
            elif lan == 'th':
                from spacy.lang.th.stop_words import STOP_WORDS
            elif lan == 'tr':
                from spacy.lang.tr.stop_words import STOP_WORDS
            elif lan == 'uk':
                from spacy.lang.uk.stop_words import STOP_WORDS
            elif lan == 'ur':
                from spacy.lang.ur.stop_words import STOP_WORDS
            elif lan == 'vi':
                from spacy.lang.vi.stop_words import STOP_WORDS
            elif lan == 'yo':
                from spacy.lang.yo.stop_words import STOP_WORDS

            return nlp, STOP_WORDS
        else:
            print(f'\033[91m\tLanguage not managed, select in this languages:\n\t{",".join(self.all_spacy_lan)}\n\t'
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
        print(
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
        Train CNN text categorizer with embeddings
        """
        from spacy.util import minibatch
        import numpy as np
        import tqdm
        categ = np.asarray(classes)
        nr_categ = len(categ)
        texts = list(texts)
        y_lab = list(y_lab)
        texts_test = list(texts_test)
        y_lab_test = list(y_lab_test)

        def __load_textcat_data(texts_, y_lab_, texts_test_, y_lab_test_):
            """
            Load and prepare text categorization data
            """
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
                        d[str(ncat)] = True
                    else:
                        d[str(ncat)] = False
                return d
            train_cats_ = [__gen_dict_label(y) for y in train_labels]
            eval_cats_ = [__gen_dict_label(y) for y in eval_labels]
            return (train_texts_, train_cats_), (eval_texts_, eval_cats_)

        opt_params = self.get_opt_params(locals())
        textcat = self.nlp.create_pipe(
            "textcat",
            config={
                "exclusive_classes": True,
                "architecture": "simple_cnn",
            }
        )

        self.nlp.add_pipe(textcat, last=True)
        for cl in categ:
            textcat.add_label(cl)

        (train_texts, train_cats), (dev_texts, dev_cats) = __load_textcat_data(texts, y_lab, texts_test, y_lab_test)
        print(
            "Number of examples ({} training, {} evaluation)".format(
                len(train_texts), len(dev_texts)
            )
        )
        train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))
        best_acc = 0.0

        class EarlyStopping(object):
            """
            Early stopping callback
            """

            def __init__(self, metric, patience_):
                self.metric = metric
                self.max_patience = patience_
                self.current_patience = patience_
                self.best = 0.5

            def update(self, result):
                """
                Update early stopping state
                """
                if result[self.metric] >= self.best:
                    self.best = result[self.metric]
                    self.current_patience = self.max_patience
                    return False
                else:
                    self.current_patience -= 1
                    return self.current_patience <= 0

        early_stopping = EarlyStopping("acc", patience)

        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "textcat"]
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            self.configure_optimizer(optimizer, opt_params)
            print("Training the model...")
            print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
            for i in range(epoch):
                losses = {"textcat": 0.0}
                if use_tqdm:
                    try:
                        train_data = tqdm.tqdm(train_data, leave=False)
                    except Exception as e:
                        print(e)
                        pass
                batches = minibatch(train_data, size=batch_size)
                for batch in batches:
                    texts, annotations = zip(*batch)
                    self.nlp.update(
                        texts, annotations, sgd=optimizer, drop=dropout, losses=losses
                    )
                with textcat.model.use_params(optimizer.averages):
                    scores = self.evaluate_textcat(self.nlp.tokenizer, textcat, dev_texts, dev_cats)
                    scores1 = self.evaluate_textcat(self.nlp.tokenizer, textcat, train_texts, train_cats)
                print('Epoca ' + str(i))
                print('\nResults Test:\n')
                best_acc = max(best_acc, scores["acc"])
                self.report_progress(i, best_acc, losses, scores)
                print('\nResults Train:\n')
                best_acc = max(best_acc, scores1["acc"])
                self.report_progress(i, best_acc, losses, scores1)
                should_stop = early_stopping.update(scores)
                if should_stop:
                    print('The model does not learn, try to change CNN architecture')
                    break

        print('Predicting values train...')
        ytrain = []
        for ytr in train_cats:
            ytrain.append(str(categ[np.where([i for prev, i in ytr.items()])[0][0]]))
        ytest = []
        for yte in dev_cats:
            ytest.append(str(categ[np.where([i for prev, i in yte.items()])[0][0]]))

        print('Predicting values test...')
        fitted_y_test = []
        for i, val in enumerate(dev_texts):
            fitted_y_test.append(categ[np.argmax([i for prev, i in self.nlp(val).cats.items()])])
        fitted_y_train = []
        for i, val in enumerate(train_texts):
            fitted_y_train.append(categ[np.argmax([i for prev, i in self.nlp(val).cats.items()])])

        print('Last things...')
        from sklearn.metrics import classification_report, confusion_matrix

        clasification_report_tr = classification_report(ytrain, fitted_y_train)
        confusion_matrix_tr = confusion_matrix(ytrain, fitted_y_train)
        clasification_report_te = classification_report(ytest, fitted_y_test)
        confusion_matrix_te = confusion_matrix(ytest, fitted_y_test)

        previsioni_scores = {'classification_report_tr': clasification_report_tr,
                             'confusion_matrix_tr': confusion_matrix_tr,
                             'classification_report_te': clasification_report_te,
                             'confusion_matrix_te': confusion_matrix_te}
        
        return self.nlp, previsioni_scores
