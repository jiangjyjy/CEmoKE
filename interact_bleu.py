import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config, BertTokenizer
from transformers import BertTokenizerFast, TextGenerationPipeline
# from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain
# from chatbot.model import DialogueGPT2Model
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import math
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
import nltk
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
from templateTool import Business
from templateTool_changebot import BusinessChangBot
from muti_templateTool import muti_Business
from xlsxTool import ExcelUtils
import jiagu
from snownlp import SnowNLP
import jieba
from snownlp import sentiment
from LAC import LAC
from bert_score import score
import openpyxl
import json
import requests
from ownthink_semantic import Analysis

import warnings
warnings.filterwarnings("ignore")

PAD = '[PAD]'
pad_id = 0

# 装载LAC模型（分词和词性标注）
lac = LAC(mode='lac')

def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0.9, type=float, required=False, help='最高积累概率')
    parser.add_argument('--model_config', default='config/config.json', type=str, required=False,
                        help='模型参数')
    parser.add_argument('--log_path', default='data/interact.log', type=str, required=False, help='interact日志存放位置')
    parser.add_argument('--vocab_path', default='vocab/vocab.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--model_path', default='model/epoch17', type=str, required=False,
                        help='对话模型路径')
    parser.add_argument('--save_samples_path', default="sample/", type=str, required=False, help="保存聊天记录的文件路径")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    # parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--max_len', type=int, default=25, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=5, help="dialogue history的最大长度")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    return parser.parse_args()


def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    return logger


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocab size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷
    
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def entityextraction(text):
    entity = []
    EE = lac.run(text)
    for i in range(len(EE[1])):
        if EE[1][i] == 'PER':
            temp = '人名: ' + EE[0][i]
            entity.append([temp])
        elif EE[1][i] == 'ORG':
            temp = '机构名: ' + EE[0][i]
            entity.append([temp])
        elif EE[1][i] == 'LOC':
            temp = '地名: ' + EE[0][i]
            entity.append([temp])
        elif EE[1][i] == 'TIME' or (EE[1][i] == 'm' and (EE[0][i][-1] == '年' or EE[0][i][-1] == '天')) or (EE[0][i][-1] == '月' and EE[0][i][-2] == '个'):
            temp = '时间: ' + EE[0][i]
            entity.append([temp])
        else:
            continue
    return entity


def main():
    args = set_args()
    logger = create_logger(args)
    
    # 处理模板对话xlsx文件
    business = Business(os.path.abspath("./templatedialog.xlsx"))
    
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    tokenizer = BertTokenizerFast(vocab_file=args.vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    # tokenizer = BertTokenizer(vocab_file=args.vocab_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    # model = transformers.AutoTokenizer.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()
    if args.save_samples_path:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/samples.txt', 'a', encoding='utf8')
        samples_file.write("聊天记录{}:\n".format(datetime.now()))
    # 存储聊天记录，每个utterance以token的id的形式进行存储
    history = []
    
    # 对话模式循环计数器
    loopCount = 0
    loopCount_templateDialog = 0
    loopCount_multi_templateDialog = 0
    print('开始和chatbot聊天，输入CTRL + Z以退出')
    
    print('请选择：\n1.单人对话       2.多人对话（完善中...）')
    dialogmode = input()

    
    while True:
        
        if dialogmode == '1' or dialogmode == '单人' or dialogmode == '单' or dialogmode == '一' or dialogmode == '单人对话' or dialogmode == 'dan' or dialogmode == 'yi' or dialogmode == 'danrenduihua' or dialogmode == 'danren':
            print("/****************************/")
            print("  欢 迎 使 用 单 人 对 话 模 式  ")
            print("/****************************/")

            # login
            # 默认bot为1号
            botno = 1
            bot = business.getBotByNo(botno)
            print("您先选择一个登录用户身份?");
            for key in business.sheet3_ouput.keys():
                # print(key)
                user = business.getUserByNo(key)
                # print(user)
                print(str(user["no"]) + "-" + str(user["username"]));
            flag = True
            no = None
            while (flag):
                print("请选择数字：");
                no = input();
                user = None
                bot = None
                try:
                    user = business.getUserByNo(int(no))
                    bot = business.getBotByNo(botno)
                except:
                    pass
                if (user != None):
                    business.user = user;
                    print(str(user["username"]) + "，您好，可以开始对话了！")
                    print(str("注意：按字母 q 退出服务喔！"))
                    flag = False
                else:
                    print("选择错误！请重新输入")
                    flag = True
                if (bot != None):
                    business.bot = bot;
                    # print(str(user["username"]) + "，您好，可以开始对话了！")
                    # print(str("注意：按字母 q 退出服务喔！"))
                    flag = False
                else:
                    print("bot出现错误！！！")
                    flag = True

            #print(business.userLogin())
            # business.beginAsk()
            while True:
                loopCount += 1
                flag_fun1 = 0
                flag_fun2 = 0
                try:
                    text = input("user:")
                    # baidu lac分词和词性标注获取
                    lac_ws_pos = lac.run(text)
                    #print(lac_ws_pos)
                    slu = Analysis(text)
                    data = slu.analysis()
                    #print(lac_ws_pos[0][-1][-1])
                    print("{'domain':", slu.domain, "; 'intent':", slu.intent, "; 'slot':", entityextraction(text), "}")
                    '''等待修...
                    for i in range(len(lac_ws_pos[1])):
                        if lac_ws_pos[1][i] == 'LOC':
                            lac_ws_pos[0][i]
                        elif lac_ws_pos[1][i] == 'PER':
                            lac_ws_pos[0][i]
                        else:
                            continue
                    '''
                    '''
                    # jiaba: Word Segmentation
                    print("jieba (word segmentation): ")
                    jieba_ws = jieba.cut(text)
                    jieba_ws = list(jieba_ws)
                    print(jieba_ws)
                    # snownlp: Word Segmentation, Part-Of-Speech Tagging, Keyword Extraction, Sentiment Analysis
                    print("snownlp (word segmentation, POS, keyword extraction, sentiment analysis): ")
                    s = SnowNLP(text)
                    print(s.words)
                    print(list(s.tags))
                    print(s.keywords(2))
                    print(s.sentiments)
                    # jiagu: Word Segmentation, Part-Of-Speech Tagging, Named Entity Recognition, Sentiment Analysis
                    print("jiagu (word segmentation, POS, NER, keyword extraction, sentiment analysis): ")
                    jiagus = jiagu.seg(text)  # 分词
                    print(jiagus)
                    print(jiagu.pos(jiagus))
                    print(jiagu.ner(jiagus))
                    print(jiagu.keywords(text, 2))
                    print(jiagu.sentiment(text))
                    # lac: Word Segmentation, Part-Of-Speech Tagging, Named Entity Recognition
                    lac = LAC(mode='seg')
                    lac_ws = lac.run(text)
                    print(lac_ws)
                    lac = LAC(mode='lac')
                    lac_pos = lac.run(text)
                    print(lac_pos)
                    '''
                    # snownlp情感分析
                    user_sentiment = SnowNLP(text)
                    if user_sentiment.sentiments > 0.45 and user_sentiment.sentiments <= 1.0:
                        text_sentiment = 'positive'
                        print("user emotion: ", end='')
                        print("\033[31m", end='')
                        print(text_sentiment, end=' ')
                        print(user_sentiment.sentiments, end='')
                        print("\033[0m")
                    elif user_sentiment.sentiments <= 0.45 and user_sentiment.sentiments >= 0.0:
                        text_sentiment = 'negative'
                        print("user emotion: ", end='')
                        print("\033[35m", end='')
                        print(text_sentiment, end=' ')
                        print(user_sentiment.sentiments, end='')
                        print("\033[0m")
                    
                    if args.save_samples_path:
                        samples_file.write("user:{}\n".format(text))
                    # text = "你好"
                    
                    # 当“说一下香港大学”，“说一下”/“讲一下”/“解释一下”/“搜索一下”即触发百度百科
                    baidubaike = lac.run(text)
                    baidubaike_text = []
                    baidubaike_judge = []
                    for bdbk in range(len(baidubaike[0])):
                        # print(i)
                        if bdbk == 0 or bdbk == 1:
                            baidubaike_judge.append(baidubaike[0][bdbk])
                        else:
                            baidubaike_text.append(baidubaike[0][bdbk])
                    baidubaike_judge = ''.join(baidubaike_judge)
                    
                    # 根据ownthink知识图谱进行百度百科的回复（不将百度百科内容和问题计入dialogue history）
                    if baidubaike_judge == '说一下' or baidubaike_judge == '讲一下' or baidubaike_judge == '解释一下' or baidubaike_judge == '搜索一下':
                        loopCount_templateDialog = 0
                        baidubaike_text = ''.join(baidubaike_text)
                        ownthink_KGBD_url = 'https://api.ownthink.com/kg/knowledge?entity=' + baidubaike_text
                        response_baidubaike = json.loads(requests.get(ownthink_KGBD_url).text)
                        if response_baidubaike['data'] == "" or response_baidubaike['data'] == {} or \
                                response_baidubaike['message'] == 'error' or response_baidubaike['data']['desc'] == "":
                            print("chatbot:抱歉，我没有理解您的意思！")
                        else:
                            print("chatbot:" + response_baidubaike['data']['desc'])
                        # print("chatbot:" + json.loads(requests.get(ownthink_KGBD_url).text))
                        judgment_multi_templatedialog = 'no'
                    
                    else:
                        # 计算history dialogue
                        # ------------------------------------------------------------------------------------ #
                        text_ids = tokenizer.encode(text, add_special_tokens=False)
                        history.append(text_ids)
                        # print(text_ids)
                        input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头
                        # print(input_ids)
                        
                        for history_id, history_utr in enumerate(history[-args.max_history_len:]):
                            input_ids.extend(history_utr)
                            input_ids.append(tokenizer.sep_token_id)
                        # print(input_ids)  #检测history的方法
                        input_ids = torch.tensor(input_ids).long().to(device)
                        input_ids = input_ids.unsqueeze(0)
                        # ------------------------------------------------------------------------------------ #
                        
                        con_text = [text]
                        # print(con_text)
                        # print([text])
                        # 处理输入的text的格式，便于bleu计算
                        updated_text = []
                        for i in range(len(con_text)):
                            updated_text = list(con_text[i])  # 调整成为bleu计算中的candidate格式
                        # print(con_text)
                        # print(updated_text)
                        
                        # 计算trigger response的数量并提取所有的trigger response
                        triggerresp = business.getAllTrigger()  # bleu
                        triggerresp_num = len(business.getAllTrigger())
                        # print(triggerresp)
                        
                        # 遍历所有的trigger response，提取最大的eval值
                        max_evaluation = 0.0
                        for j in range(triggerresp_num):
                            # con_triggerresp = [triggerresp[j]]  # 调整成为bleu的reference格式
                            con_triggerresp = ''.join(triggerresp[j])
                            # 调整为计算bleu的candidate和reference的格式，并且可以将插件单独提取出来
                            bleu_candidate = business.format(text)
                            bleu_reference = business.format(con_triggerresp)  # 支持将trigger response的插件单独提取出来
                            # 将reference中的插件部分去掉（防止识别上的误差）
                            bleu_reference_updated = []
                            for pluginval in range(len(bleu_reference)):
                                if len(bleu_reference[pluginval]) > 1:
                                    continue
                                bleu_reference_updated.append(bleu_reference[pluginval])


                            function1 = ['我', '想', '把', '你', '的', '名', '字', '改', '成']
                            function2 = ['我', '记', '得', '我', '之', '前', '在', '工', '作', '过']
                            flag_tr_pi = 0
                            if sentence_bleu([function1], bleu_candidate, weights=(1, 0, 0, 0)) >= 0.8:
                                flag_tr_pi = 1
                            elif sentence_bleu([function2], bleu_candidate, weights=(1, 0, 0, 0)) > 0.8:
                                flag_tr_pi = 2

                            # 功能一：改机器人的名字ft_changebotname：将candidate中与reference平行的插件去掉（防止识别误差）
                            # 触发条件（我想把你的名字改成...）
                            trigger_ft_changebotname = []
                            if len(bleu_candidate) > 8 and flag_tr_pi == 1:
                                flag_tr_pi = 0
                                # 这段for循环没有意义
                                for trigger_fun1 in range(len(bleu_candidate)):
                                    #print(len(bleu_candidate))
                                    if trigger_fun1 == 0 or trigger_fun1 == 1 or trigger_fun1 == 2 or trigger_fun1 == 3 or \
                                            trigger_fun1 == 4 or trigger_fun1 == 5 or trigger_fun1 == 6 or trigger_fun1 == 7 or \
                                            trigger_fun1 == 8:
                                        trigger_ft_changebotname.append(bleu_candidate[trigger_fun1])
                                    else:
                                        break
                                if bleu_reference_updated == ['我', '想', '把', '你', '的', '名', '字', '改', '成']:
                                    #print("1111")
                                    lac_candidate = lac.run(text)
                                    for k in range(len(lac_candidate[1])):
                                        if lac_candidate[1][k] == 'PER' or lac_candidate[1][k] == 'an':
                                            # 注意需要修改，如果有多个PER，则顺序不同
                                            botPER_updated = lac_candidate[0][k]
                                            del lac_candidate[0][k]
                                        # elif lac_candidate[1][k] == 'LOC':
                                        #     del lac_candidate[0][k]
                                        else:
                                            continue
                                    ft_changebotname = ''.join(lac_candidate[0])
                                    bleu_candidate_updated_ft_changebotname = business.format(ft_changebotname)
                                    if ft_changebotname == '我想把你的名字改成':
                                        if botPER_updated != business.bot["pbotname"]:
                                            bleu_candidate = bleu_candidate_updated_ft_changebotname
                                            wb = openpyxl.load_workbook(
                                                './templatedialog.xlsx')
                                            sheet = wb['bot_information']
                                            sheet['E2'] = botPER_updated
                                            wb.save('templatedialog.xlsx')
                                        elif botPER_updated == business.bot["pbotname"]:
                                            flag_fun1 = 1
                                            response = "我现在的名字就是" + business.bot["pbotname"] + "哦，可以给我改个不同的名字哈"
                                            response_ids = tokenizer.encode(response, add_special_tokens=False)
                                            history.append(response_ids)
                                            print("chatbot:" + response)
                                            judgment_multi_templatedialog = 'no'
                                            loopCount_templateDialog = 0
                                            break
                                        business = BusinessChangBot("./templatedialog.xlsx")
                                        userno = no
                                        business.userLogin(userno)

                            '''
                            DST，FST看一下
                            体现出来——网页端（FYP）
                            匹配
                            DST
                            emotional dialogue
                            论文笔记
                            '''
                            # 功能二：识别工作地点：将candidate中与reference平行的插件去掉（防止识别误差）
                            trigger_ft_judgejobworkplace = []
                            if len(bleu_candidate) > 10 and flag_tr_pi == 2:
                                flag_tr_pi = 0
                                for trigger_fun2 in range(len(bleu_candidate)):
                                    #print(len(bleu_candidate))
                                    if trigger_fun2 == 0 or trigger_fun2 == 1 or trigger_fun2 == 2 or trigger_fun2 == 3 or \
                                            trigger_fun2 == 4 or trigger_fun2 == 5 or trigger_fun2 == 6 or trigger_fun2 == 8 or \
                                            trigger_fun2 == 9 or trigger_fun2 == 10:
                                        trigger_ft_judgejobworkplace.append(bleu_candidate[trigger_fun2])
                                        #print("111")
                                    elif trigger_fun2 == 7:
                                        del bleu_candidate[trigger_fun2]
                                        #print("sss")
                                    else:
                                        break
                                # print(trigger_ft_judgejobworkplace)
                                if trigger_ft_judgejobworkplace == ['我', '记', '得', '我', '之', '前', '在', '工', '作', '过']:
                                    # print("1111111111111111111111111")
                                    lac_candidate_2 = lac.run(text)
                                    for k in range(len(lac_candidate_2[1])):
                                        if lac_candidate_2[1][k] == 'LOC' or lac_candidate_2[1][k] == 'ORG':
                                            jobworkplace = lac_candidate_2[0][k]
                                            del lac_candidate_2[0][k]
                                        else:
                                            continue
                                    ft_judgejobworkplace = ''.join(lac_candidate_2[0])
                                    bleu_candidate_updated_ft_judgejobworkplace = business.format(ft_judgejobworkplace)
                                    if jobworkplace != business.user["joblocation"]:
                                        # print("2222222222222222222222")
                                        flag_fun2 = 1
                                        response = "我记得您之前是在" + business.user["joblocation"] + "工作哈"
                                        response_ids = tokenizer.encode(response, add_special_tokens=False)
                                        history.append(response_ids)
                                        print("chatbot:" + response)
                                        judgment_multi_templatedialog = 'no'
                                        loopCount_templateDialog = 0
                                        break

                            # 计算bleu的值

                            '''
                            if bleu_reference_updated == ['我', '之', '前', '在', '工', '作', '过']:
                                evaluation = sentence_bleu([bleu_reference_updated], bleu_candidate_updated_ft_judgejobworkplace,
                                                           weights=(1, 0, 0, 0))
                            elif bleu_reference_updated == ['我', '想', '把', '你', '的', '名', '字', '改', '成']:
                                evaluation = sentence_bleu([bleu_reference_updated], bleu_candidate_updated_ft_changebotname,
                                                           weights=(1, 0, 0, 0))
                            '''
                            evaluation = sentence_bleu([bleu_reference_updated], bleu_candidate, weights=(1, 0, 0, 0))
                            # temporary = evaluation
                            # print(evaluation)
                            if evaluation >= max_evaluation:
                                max_evaluation = evaluation
                            else:
                                max_evaluation = max_evaluation
                        # print("111111111111111")
                        '''
                        判断是否进行多轮模板对话
                        '''
                        # 初始化next_template的值（yes or no）
                        if loopCount == 1:
                            judgment_multi_templatedialog = 'no'
                        if judgment_multi_templatedialog == 'no':
                            loopCount_multi_templateDialog = 0
                        
                        # 提前判定是否继续走多轮模板对话
                        # print(loopCount_multi_templateDialog)
                        if loopCount_multi_templateDialog == 0 and judgment_multi_templatedialog == 'yes':
                            # judgment0 = business.getNoByTrigger(text)
                            judgment1 = business.getNoMulti1ByTrigger(text, no_multi_templatedialog)
                            # print(judgment1)
                            if judgment1 == None:
                                judgment_multi_templatedialog = 'no'
                            else:
                                judgment_multi_templatedialog = 'yes'
                        elif loopCount_multi_templateDialog == 1 and judgment_multi_templatedialog == 'yes':
                            judgment2 = business.getNoMulti2ByTrigger(text, no_multi_templatedialog,
                                                                      nomt1_multi_templatedialog)
                            # print(judgment2)
                            if judgment2 == None:
                                judgment_multi_templatedialog = 'no'
                            else:
                                judgment_multi_templatedialog = 'yes'
                        elif loopCount_multi_templateDialog == 2 and judgment_multi_templatedialog == 'yes':
                            judgment3 = business.getNoMulti3ByTrigger(text, no_multi_templatedialog,
                                                                      nomt1_multi_templatedialog,
                                                                      nomt2_multi_templatedialog)
                            # print(judgment2)
                            if judgment3 == None:
                                judgment_multi_templatedialog = 'no'
                            else:
                                judgment_multi_templatedialog = 'yes'
                        elif loopCount_multi_templateDialog == 3 and judgment_multi_templatedialog == 'yes':
                            judgment4 = business.getNoMulti4ByTrigger(text, no_multi_templatedialog,
                                                                      nomt1_multi_templatedialog,
                                                                      nomt2_multi_templatedialog,
                                                                      nomt3_multi_templatedialog)
                            # print(judgment2)
                            if judgment4 == None:
                                judgment_multi_templatedialog = 'no'
                            else:
                                judgment_multi_templatedialog = 'yes'
                        elif loopCount_multi_templateDialog == 4 and judgment_multi_templatedialog == 'yes':
                            judgment5 = business.getNoMulti5ByTrigger(text, no_multi_templatedialog,
                                                                      nomt1_multi_templatedialog,
                                                                      nomt2_multi_templatedialog,
                                                                      nomt3_multi_templatedialog,
                                                                      nomt4_multi_templatedialog)
                            # print(judgment2)
                            if judgment5 == None:
                                judgment_multi_templatedialog = 'no'
                            else:
                                judgment_multi_templatedialog = 'yes'
                        
                        # 判断是走下一段模板对话还是“从头开始模板对话orGPT2生成对话”
                        if judgment_multi_templatedialog == 'yes':
                            # 多轮模板对话计数器
                            loopCount_multi_templateDialog += 1
                            # 直接走下一段模板对话
                            if loopCount_multi_templateDialog == 1:
                                response = business.beginAnswer_Multi1(text, no_multi_templatedialog)
                            elif loopCount_multi_templateDialog == 2:
                                response = business.beginAnswer_Multi2(text, no_multi_templatedialog,
                                                                       nomt1_multi_templatedialog)
                            elif loopCount_multi_templateDialog == 3:
                                response = business.beginAnswer_Multi3(text, no_multi_templatedialog,
                                                                       nomt1_multi_templatedialog,
                                                                       nomt2_multi_templatedialog)
                            elif loopCount_multi_templateDialog == 4:
                                response = business.beginAnswer_Multi4(text, no_multi_templatedialog,
                                                                       nomt1_multi_templatedialog,
                                                                       nomt2_multi_templatedialog,
                                                                       nomt3_multi_templatedialog)
                            elif loopCount_multi_templateDialog == 5:
                                response = business.beginAnswer_Multi5(text, no_multi_templatedialog,
                                                                       nomt1_multi_templatedialog,
                                                                       nomt2_multi_templatedialog,
                                                                       nomt3_multi_templatedialog,
                                                                       nomt4_multi_templatedialog)
                            else:
                                print("loopCount_multi_templateDialog出现错误！！！")
                                response = '错误！！！'
                                judgment_multi_templatedialog = 'no'
                            
                            response_ids = tokenizer.encode(response, add_special_tokens=False)
                            history.append(response_ids)
                            print("chatbot:" + response)
                            
                            # 获取是否有下一次模板对话
                            if loopCount_multi_templateDialog == 1:
                                nomt1_multi_templatedialog = business.getNoMulti1ByTrigger(text,
                                                                                           no_multi_templatedialog)
                                judgment_multi_templatedialog = business.getNextTemplateByNo_Multi2(
                                    no_multi_templatedialog, nomt1_multi_templatedialog)
                                # print(no_multi_templatedialog)
                                # print(nomt1_multi_templatedialog)
                                # print(judgment_multi_templatedialog)
                            elif loopCount_multi_templateDialog == 2:
                                nomt2_multi_templatedialog = business.getNoMulti2ByTrigger(text,
                                                                                           no_multi_templatedialog,
                                                                                           nomt1_multi_templatedialog)
                                judgment_multi_templatedialog = business.getNextTemplateByNo_Multi3(
                                    no_multi_templatedialog, nomt1_multi_templatedialog, nomt2_multi_templatedialog)
                                # print(no_multi_templatedialog)
                                # print(nomt1_multi_templatedialog)
                                # print(nomt2_multi_templatedialog)
                                # print(judgment_multi_templatedialog)
                            elif loopCount_multi_templateDialog == 3:
                                nomt3_multi_templatedialog = business.getNoMulti3ByTrigger(text,
                                                                                           no_multi_templatedialog,
                                                                                           nomt1_multi_templatedialog,
                                                                                           nomt2_multi_templatedialog)
                                judgment_multi_templatedialog = business.getNextTemplateByNo_Multi4(
                                    no_multi_templatedialog, nomt1_multi_templatedialog, nomt2_multi_templatedialog,
                                    nomt3_multi_templatedialog)
                            elif loopCount_multi_templateDialog == 4:
                                nomt4_multi_templatedialog = business.getNoMulti4ByTrigger(text,
                                                                                           no_multi_templatedialog,
                                                                                           nomt1_multi_templatedialog,
                                                                                           nomt2_multi_templatedialog,
                                                                                           nomt3_multi_templatedialog)
                                judgment_multi_templatedialog = business.getNextTemplateByNo_Multi5(
                                    no_multi_templatedialog, nomt1_multi_templatedialog, nomt2_multi_templatedialog,
                                    nomt3_multi_templatedialog, nomt4_multi_templatedialog)
                            elif loopCount_multi_templateDialog == 5:
                                nomt5_multi_templatedialog = business.getNoMulti5ByTrigger(text,
                                                                                           no_multi_templatedialog,
                                                                                           nomt1_multi_templatedialog,
                                                                                           nomt2_multi_templatedialog,
                                                                                           nomt3_multi_templatedialog,
                                                                                           nomt4_multi_templatedialog)
                                judgment_multi_templatedialog = business.getNextTemplateByNo_Multi6(
                                    no_multi_templatedialog, nomt1_multi_templatedialog, nomt2_multi_templatedialog,
                                    nomt3_multi_templatedialog, nomt4_multi_templatedialog, nomt5_multi_templatedialog)
                            else:
                                print("loopCount_multi_templateDialog出现错误！！！")
                                judgment_multi_templatedialog = 'no'
                            continue
                        
                        elif judgment_multi_templatedialog == 'no':
                            judgment_multi_templatedialog = 'no'
                        
                        else:
                            print('yes出现错误！！！')
                            judgment_multi_templatedialog = 'no'
                        
                        if judgment_multi_templatedialog == 'no' or judgment_multi_templatedialog == None:
                            # 如果为no，后续多轮模板对话计数器立刻归0
                            # print("2222222222222222222222222222")
                            loopCount_multi_templateDialog = 0
                            # 判断走模板对话还是GPT2
                            # 如果bleu大于或者等于80%则使用模板回复

                            if max_evaluation >= 0.8 and flag_fun1 == 0 and flag_fun2 == 0:
                                #print("3333333333333333")
                                loopCount_templateDialog = 0
                                loopCount_templateDialog = loopCount_templateDialog + 1
                                response = business.beginAnswer(text)
                                # print("3333333333333333333333333333")
                                # input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
                                response_ids = tokenizer.encode(response, add_special_tokens=False)
                                # response_ids.append(tokenizer.sep_token_id)
                                history.append(response_ids)
                                
                                print("chatbot:" + response)
                                # 获取是否有下一次模板对话
                                no_multi_templatedialog = business.getNoByTrigger(text)
                                judgment_multi_templatedialog = business.getNextTemplateByNo_Multi1(
                                    no_multi_templatedialog)
                                # print(no_multi_templatedialog)
                                # print(judgment_multi_templatedialog)
                                
                                if args.save_samples_path:
                                    samples_file.write("chatbot:{}\n".format("".join(response)))
                            
                            # 如果bleu小于80%则使用GPT2进行兜底自动回复
                            elif max_evaluation >= 0 and max_evaluation < 0.8 and flag_fun1 == 0 and flag_fun2 == 0:
                                loopCount_templateDialog = 0
                                response = []  # 根据context，生成的response
                                # 最多生成max_len个token
                                for _ in range(args.max_len):
                                    outputs = model(input_ids=input_ids)
                                    logits = outputs.logits
                                    next_token_logits = logits[0, -1, :]
                                    # print(next_token_logits)
                                    # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                                    for id in set(response):
                                        next_token_logits[id] /= args.repetition_penalty
                                    next_token_logits = next_token_logits / args.temperature
                                    
                                    # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                                    next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                                    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk,
                                                                            top_p=args.topp)
                                    # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                                    if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                                        break
                                    response.append(next_token.item())
                                    input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
                                    # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
                                    # print("his_text:{}".format(his_text))
                                history.append(response)
                                # print(history)
                                text = tokenizer.convert_ids_to_tokens(response)
                                print("chatbot:" + "".join(text))
                                
                                # 如果走了GPT2，默认模板对话后续为no
                                judgment_multi_templatedialog = 'no'
                                
                                if args.save_samples_path:
                                    samples_file.write("chatbot:{}\n".format("".join(text)))
                        
                        elif judgment_multi_templatedialog == 'yes':
                            continue
                        
                        else:
                            print("judgment_multi_templatedialog出现问题！！！")
                            response = '错误！！！'
                            print("chatbot:" + response)
                            judgment_multi_templatedialog = 'no'
                        
                        '''
                        # 如果bleu大于或者等于80%则使用模板回复
                        if max_evaluation >= 0.8:
                            response = business.beginAnswer(text)
                            # input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
                            response_ids = tokenizer.encode(response, add_special_tokens=False)
                            # response_ids.append(tokenizer.sep_token_id)
                            history.append(response_ids)
                            print("chatbot:" + response)
                            if args.save_samples_path:
                                samples_file.write("chatbot:{}\n".format("".join(response)))

                        # 如果bleu小于80%则使用GPT2进行兜底自动回复
                        else:
                            response = []  # 根据context，生成的response
                            # 最多生成max_len个token
                            for _ in range(args.max_len):
                                outputs = model(input_ids=input_ids)
                                logits = outputs.logits
                                next_token_logits = logits[0, -1, :]
                                # print(next_token_logits)
                                # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                                for id in set(response):
                                    next_token_logits[id] /= args.repetition_penalty
                                next_token_logits = next_token_logits / args.temperature

                                # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                                next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
                                # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                                if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                                    break
                                response.append(next_token.item())
                                input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
                                # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
                                # print("his_text:{}".format(his_text))
                            history.append(response)
                            # print(history)
                            text = tokenizer.convert_ids_to_tokens(response)
                            print("chatbot:" + "".join(text))
                            if args.save_samples_path:
                                samples_file.write("chatbot:{}\n".format("".join(text)))
                        '''
                
                except KeyboardInterrupt:
                    if args.save_samples_path:
                        samples_file.close()
        
        
        elif dialogmode == '2' or dialogmode == '多人' or dialogmode == '多' or dialogmode == '二' or dialogmode == '多人对话' or dialogmode == 'er' or dialogmode == 'duo' or dialogmode == 'duorenduihua' or dialogmode == 'duoren':
            if loopCount == 1:
                print("/****************************/")
                print("  欢 迎 使 用 多 人 对 话 模 式  ")
                print("/****************************/")
                print("备注：完善中......")
            
            flag = True
            
            while (flag):
                try:
                    mutibusiness = muti_Business("C:/Users/99487/Desktop/templatedialog/templatedialog.xlsx")
                    mutibusiness.userLogin()
                    # username = mutibusiness.getUserByNo()
                    text = input(">> ")
                    if args.save_samples_path:
                        samples_file.write("user:{}\n".format(text))
                    # text = "你好"
                    # 计算history dialogue
                    # ------------------------------------------------------------------------------------ #
                    text_ids = tokenizer.encode(text, add_special_tokens=False)
                    history.append(text_ids)
                    # print(text_ids)
                    input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头
                    # print(input_ids)
                    
                    for history_id, history_utr in enumerate(history[-args.max_history_len:]):
                        input_ids.extend(history_utr)
                        input_ids.append(tokenizer.sep_token_id)
                    # print(input_ids)
                    input_ids = torch.tensor(input_ids).long().to(device)
                    input_ids = input_ids.unsqueeze(0)
                    # ------------------------------------------------------------------------------------ #
                    
                    con_text = [text]
                    # print([text])
                    # 处理输入的text的格式，便于bleu计算
                    updated_text = []
                    for i in range(len(con_text)):
                        updated_text = list(con_text[i])
                    # print(con_text)
                    # print(updated_text)
                    
                    # 计算trigger response的数量并提取所有的trigger response
                    triggerresp = mutibusiness.getAllTrigger()
                    triggerresp_num = len(mutibusiness.getAllTrigger())
                    
                    # 遍历所有的trigger response，提取最大的eval值
                    max_evaluation = 0.0
                    for j in range(triggerresp_num):
                        con_triggerresp = [triggerresp[j]]
                        evaluation = sentence_bleu(con_triggerresp, updated_text, weights=(1, 0, 0, 0))
                        # temporary = evaluation
                        if evaluation >= max_evaluation:
                            max_evaluation = evaluation
                        else:
                            max_evaluation = max_evaluation
                        # con_temporary = max_evaluation
                    
                    # 如果bleu大于或者等于90%则使用模板回复
                    if max_evaluation >= 0.8:
                        response = mutibusiness.beginAnswer(text)
                        response_ids = tokenizer.encode(response, add_special_tokens=False)
                        # response_ids.append(tokenizer.sep_token_id)
                        history.append(response_ids)
                        print("chatbot:" + response)
                        if args.save_samples_path:
                            samples_file.write("chatbot:{}\n".format("".join(response)))
                        break
                    
                    # 如果bleu小于90%则使用GPT2进行兜底自动回复
                    else:
                        response = []  # 根据context，生成的response
                        # 最多生成max_len个token
                        for _ in range(args.max_len):
                            outputs = model(input_ids=input_ids)
                            logits = outputs.logits
                            next_token_logits = logits[0, -1, :]
                            # print(next_token_logits)
                            # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                            for id in set(response):
                                next_token_logits[id] /= args.repetition_penalty
                            next_token_logits = next_token_logits / args.temperature
                            
                            # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
                            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
                            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
                            # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
                            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                            if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                                break
                            response.append(next_token.item())
                            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
                            # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
                            # print("his_text:{}".format(his_text))
                        history.append(response)
                        # print(history)
                        text = tokenizer.convert_ids_to_tokens(response)
                        print("chatbot:" + "".join(text))
                        if args.save_samples_path:
                            samples_file.write("chatbot:{}\n".format("".join(text)))
                        continue
                
                except KeyboardInterrupt:
                    if args.save_samples_path:
                        samples_file.close()
        
        else:
            if loopCount > 5:
                print('异常！请您在仔细阅读智能对话系统操作手册后再进行操作哦~')
                break
            else:
                print('输入不正确！请您再输入一遍吧。')
                continue


if __name__ == '__main__':
    main()
