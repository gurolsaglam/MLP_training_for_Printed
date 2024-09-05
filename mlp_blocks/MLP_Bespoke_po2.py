#Script created by Gurol Saglam (or Guerol Saglam, guerol.saglam@kit.edu, gurol.saglam@ozu.edu.tr)
#It would be great if any developers could also add their name and contacts here:

#Python native libraries
import sys
# import os
from copy import deepcopy
from math import log2

#Python public packages
# import pandas as pd
import numpy as np
from numpy import binary_repr
from sklearn.metrics import accuracy_score

#Our packages
from qat_blocks.fxpconverter import *

class MLP_Bespoke_po2:
    def __init__(self, weights, biases, fxp_inputs, fxp_qrelu, fxp_weights, fxp_biases,
                 minclass, numofClasses=2, last_layer="relu"):
        self.QX = [(fxp_inputs[0].s + fxp_inputs[0].int, fxp_inputs[0].frac)]
        self.QA = [(fxp_qrelu[0].s + fxp_qrelu[0].int, fxp_qrelu[0].frac), (fxp_qrelu[1].s + fxp_qrelu[1].int, fxp_qrelu[1].frac)]
        self.QW = [(fxp_weights[0].s + fxp_weights[0].int, fxp_weights[0].frac), (fxp_weights[1].s + fxp_weights[1].int, fxp_weights[1].frac)]
        self.QB = [(fxp_biases[0].s + fxp_biases[0].int, fxp_biases[0].frac), (fxp_biases[1].s + fxp_biases[1].int, fxp_biases[1].frac)]
        self.YMIN = minclass
        self.numofClasses = numofClasses
        self.last_layer = last_layer
        self.QI = [qi for qi in self.QX]
        self.QI.extend(self.QA)
        #print(weights)
        #print(biases)
        self.coefs = self.convertCoef(weights)
        self.intercept = self.convertIntercepts(biases)
        #print(self.coefs)
        #print(self.intercept)
    
    def get_coefs(self):
        return deepcopy(self.coefs)
    
    def get_intercepts(self):
        return deepcopy(self.intercept)
    
    def convertCoef(self, mdata):
        transp = [np.transpose(i).tolist() for i in mdata]
        coefficients = []
        for i, l in enumerate(transp):
            newn = list()
            e = self.QW[i][1]
            for n in l:
                neww=list()
                for w in n:
                    neww.append(to_fixed(w,e))
                newn.append(neww)
            coefficients.append(newn)
        return coefficients
    
    def convertIntercepts(self, mdata):
        transp = [np.transpose(i).tolist() for i in mdata]
        intercepts = []
        for i, l in enumerate(transp):
            neww = list()
            e = self.QB[i][1]
            for w in l:
                neww.append(to_fixed(w,e))
            intercepts.append(neww)
        return intercepts
    
    #Takes multiple inputs "X_test" and converts them to the desired format.
    def convertInputs(self, X_test):
        converted_X = []
        for i, l in enumerate(X_test):
            newl = []
            #if needed to convert inputs to fxp
            for a in l:
                ia = to_fixed(a,self.QX[0][1])
                newl.append(ia)
            converted_X.append(newl)
        return np.asarray(converted_X)
    
    def calc_accuracy(self, pred, y_test):
        pred = pred+self.YMIN
        return accuracy_score(pred, y_test)
    
    def get_accuracy(self, X_test, y_test):
        pred = self.predict(X_test)
        pred = pred+self.YMIN
        return accuracy_score(pred, y_test)
    
    #Predict the results with the "exact" method, no approximation.
    def predict(self, X_test):
        prediction = []
        for i, l in enumerate(X_test):
            newl = []
            #if needed to convert inputs to fxp
            for a in l:
                ia = to_fixed(a,self.QX[0][1])
                newl.append(ia)
            prediction.append(self.predict_one(newl))
        return np.asarray(prediction)
    
    def predict_one(self, x):
        inp = x
        layer = 0
        out = []
        for layer in range(len(self.coefs)):
            product_M = self.QI[layer][1] + self.QW[layer][1]
            scale_bias = product_M - self.QB[layer][1]
            relu_M = self.QA[layer][1]
            relu_I = self.QA[layer][0]
            scale_sum = 2**(product_M-relu_M)
            relu_max = 2**(relu_M + relu_I)-1
            for i,neuron in enumerate(self.coefs[layer]):
                temp = self.intercept[layer][i] * (2**scale_bias)
                if (neuron is not None): #New if line for "weight pruned neurons"
                    for j in range(len(neuron)):
                        temp += neuron[j] * inp[j]
                #print("0", temp)
                if (temp < 0):
                    temp = 0
                else:
                    if (layer != len(self.coefs) -1):
                        temp = temp//scale_sum
                        #print("1", temp)
                        if temp > relu_max:
                            temp = relu_max
                #print("2",temp)
                out.append(temp)
            inp = list(out)
            out = []
        return np.argmax(inp)
    
    #Takes multiple inputs "X_test" and calculates the output of the layer given by "last_layer"
    #"X_test" is an array of the first inputs to the MLP.
    def predict_onelayer(self, X_test, last_layer):
        prediction = []
        for i, l in enumerate(X_test):
            newl = []
            #if needed to convert inputs to fxp
            for a in l:
                ia = to_fixed(a,self.QX[0][1])
                newl.append(ia)
            prediction.append(self.predict_one_onelayer(newl, last_layer))
        # print("ret")
        # print(prediction)
        return np.asarray(prediction)
    
    #Takes one input "x" and calculates the output of the layer given by "last_layer"
    #"x" is one of the the first inputs to the MLP.
    def predict_one_onelayer(self, x, last_layer):
        inp = x
        out = []
        for layer in range(0, last_layer):
            product_M = self.QI[layer][1] + self.QW[layer][1]
            scale_bias = product_M - self.QB[layer][1]
            relu_M = self.QA[layer][1]
            relu_I = self.QA[layer][0]
            scale_sum = 2**(product_M-relu_M)
            relu_max = 2**(relu_M + relu_I)-1
            for i,neuron in enumerate(self.coefs[layer]):
                temp = self.intercept[layer][i] * (2**scale_bias)
                if (neuron is not None): #New if line for "weight pruned neurons"
                    for j in range(len(neuron)):
                        temp += neuron[j] * inp[j]
                        # print("temp")
                        # print(temp)
                #print("0", temp)
                if (temp < 0):
                    temp = 0
                else:
                    if (layer != len(self.coefs) -1):
                        temp = temp//scale_sum
                        #print("1", temp)
                        if temp > relu_max:
                            temp = relu_max
                #print("2",temp)
                out.append(temp)
            # print("out")
            # print(out)
            inp = list(out)
            out = []
        return inp
    
    #DO NOT USE
    #These 2 functions are there to only get the maximum number of bits there will be for the Argmax.
    def no_write_neuron_comb(self, nprefix, layer, neuron, nmasks, activation, inputs, merge_list):
        bias = self.intercept[layer][neuron]
        weights = [weight for weight in self.coefs[layer][neuron]]
        
        width_inp=sum(self.QX[layer])
        decimal_w = self.QW[layer][1]
        decimal_b = self.QB[layer][1]
        decimal_i=self.QX[layer][1]
        
        fixb=(decimal_w+decimal_i)-decimal_b
        if decimal_b > (decimal_w + decimal_i):
           decimal_b = decimal_w + decimal_i
        fixb=(decimal_w+decimal_i)-decimal_b
        
        
        max_pos=0
        max_neg=0
        if bias != 0:
            b=abs(bias)*(2**fixb)
            if bias > 0:
                max_pos+=b
            else:
                max_neg+=b

        for i in range(len(weights)):
            w = weights[i]
            if w == 0:
                continue
            bit_h=get_width(abs(w))+width_inp
            
            max_prod=int(2**bit_h-1)
            if w > 0:
                max_pos+=max_prod
            else:
                max_neg+=max_prod

        spwidth=get_width(max_pos)
        snwidth=get_width(max_neg)
        swidth=max(spwidth,snwidth)+1
        decimal_s=decimal_w+decimal_i
        
        #fix relu dimensions
        rwidth = sum(self.QA[layer])
        decimal_r = rwidth-self.QA[layer][0]
        if rwidth == 32: 
            decimal_r=decimal_s
            if activation=="linear":
                rwidth=swidth
            else:
                rwidth=swidth-1
        ####
        return rwidth
    
    #DO NOT USE
    #These 2 functions are there to only get the maximum number of bits there will be for the Argmax.
    def no_write_verilog_comb(self, sum_relu_size, neuron_masks=None, order=None, arg_masks=None):
        INPNAME = "inp"
        OUTNAME = "out"
        width_a = self.QX[0][1]
        weights = self.coefs
        biases = self.intercept
        
        REGRESSOR=False
        inp_num=len(weights[0][0])
        
        width_o=int(len(weights[-1])).bit_length()
        
        act_next=[]
        act_next_size=[]
        for i in range(inp_num):
            a=INPNAME+"["+str((i+1)*width_a-1)+":"+str(i*width_a)+"]"
            act_next.append(a)
        
        ver_relu_size=0
        for j in range(len(weights)):
            act=list(act_next)
            act_next=[]
            act_next_size=[]
            for i in range(len(weights[j])):
                prefix = "n_"+str(j)+"_"
                nweights=weights[j][i]
                bias=biases[j][i]
                merge_list= [-1] * len(weights[j][i])
                for k in range(len(weights[j][i])):
                    for ii in range(i):
                        if abs(weights[j][i][k]) == abs(weights[j][ii][k]):
                            merge_list[k]=ii
                            break
                if j == len(weights)-1:
                    activation = self.last_layer
                else:
                    activation = "relu"
                #SET EMPTY MERGE LIST FOR NOW
                merge_list=[-1] * len(weights[j][i])
                if neuron_masks == None:
                    nmasks=[]
                else:
                    nmasks=neuron_masks[j][i]
                temp = self.no_write_neuron_comb(prefix, j, i, nmasks, activation, act, merge_list)
                ver_relu_size=max(ver_relu_size, temp)
                prefix=prefix+str(i)
                act_next.append(prefix)
        
        vw=max(sum_relu_size[len(weights)-1][1])
        if vw ==32:
           vw=ver_relu_size
        return vw
    
    def write_neuron_comb(self, nprefix, layer, neuron, nmasks, activation, inputs, merge_list, sum_relu_size, lenMaxVal=None):
        bias = self.intercept[layer][neuron]
        weights = [weight for weight in self.coefs[layer][neuron]]
        
        prefix = nprefix + str(neuron)
        sumname = prefix + "_sum"
        sumname_pos = sumname + "_pos"
        sumname_neg = sumname + "_neg"
        
        width_inp=sum(self.QI[layer])
        decimal_w = self.QW[layer][1]
        decimal_b = self.QB[layer][1]
        decimal_i=self.QI[layer][1]
        
        fixb=(decimal_w+decimal_i)-decimal_b
        if decimal_b > (decimal_w + decimal_i):
           decimal_b = decimal_w + decimal_i
        fixb=(decimal_w+decimal_i)-decimal_b
        
        count_neg_w=0
        pos=[]
        neg=[]

        max_pos=0
        max_neg=0

        if bias != 0:
            #print(decimal_w+decimal_i)
            #print(bias)
            b=abs(bias)<<fixb
            #print(abs(bias), fixb, b)
            width_b=get_width(b)
            bin_b=str(width_b)+"'b"+binary_repr(b,width_b)
            if bias > 0:
                pos.append(bin_b)
                max_pos+=b
            else:
                neg.append(bin_b)
                max_neg+=b

        for i in range(len(weights)):
            w = weights[i]
            if w == 0:
                print("    //weight %d : skip" % (w))
                continue
            a=inputs[i]
            name=prefix+"_po_"+str(i)
            #bit_h,bit_l=prod_width[i]
            #pwidth=bit_h
            width_w=get_width(abs(w))
            pwidth=width_w+width_inp
            bit_h=pwidth
            bin_w=str(width_w)+"'b"+binary_repr(abs(w),width_w)
            
            if nmasks:
                # quit()
                aname=name+"_a"
                print("    wire [%d:0] %s;" % ((width_inp-1),aname))
                print("    assign %s = %s;" % (aname,a))
                ones=get_ones(nmasks[i])
                print("    //mask "+str(nmasks[i])+" : "+str(ones))
                if ones:
                    if SHIFT_OR_CONCAT:
                        a=""
                        for m in get_ones(nmasks[i]):
                            if m == 0:
                                a=a+aname+"["+str(m)+"] + "
                            else:
                                a=a+"("+aname+"["+str(m)+"] << "+str(m)+") + "
                        a=a[:-3]
                    else:
                        a="}"
                        m=0
                        for abit in get_ones(nmasks[i]):
                            if abit == 0:
                                a=",1'b0"+a
                            else:
                                a=","+aname+"["+str(m)+"]"+a
                            m+=1
                        a="{"+a[1:]                    
                else:
                    a=str(width_inp)+"'d0"
            
            if w > 0:
                print("    //weight %d : %s" % (w, bin_w))
            else:
                print("    //weight abs(%d) : %s" % (w, bin_w))
            print("    wire [%d:0] %s;" % ((pwidth-1),name))
            
            if merge_list[i] < 0:
                if abs(w) == 1:
                    print("    assign %s = (%s);" % (name,a))
                else:
                    print("    assign %s = (%s) << %d;" % (name,a,log2(abs(w))))
                #print("    assign %s = $unsigned(%s) * $unsigned(%s);" % (name,a,bin_w))
            else:
                print("    //merging with node %d" %(merge_list[i]))
                mergename=nprefix+str(merge_list[i])+"_po_"+str(i)
                print("    assign %s = %s;" % (name,mergename))

            max_prod=int(2**bit_h-1)
            if w > 0:
                pos.append(name)
                max_pos+=max_prod
            else:
                neg.append(name)
                max_neg+=max_prod
            print()

        spwidth=get_width(max_pos)
        snwidth=get_width(max_neg)
        swidth=max(spwidth,snwidth)+1
        decimal_s=decimal_w+decimal_i
        
        #fix relu dimensions
        rwidth=sum_relu_size[1][0]
        decimal_r=rwidth-sum_relu_size[1][1]
        if rwidth == 32: 
            decimal_r=decimal_s
            if activation=="linear":
                rwidth=swidth
            else:
                rwidth=swidth-1
        fixrwidth=0
        if decimal_r > decimal_s:
            fixrwidth=decimal_r-decimal_s
        ####
        
        print("    //accumulate positive/negative subproducts")
        if len(pos):
            pos_str=" + ".join(str(x) for x in pos)
            print("    wire [%d:0] %s;" % ((swidth-2),sumname_pos))
            print("    assign %s = %s;" % (sumname_pos,pos_str))

        if len(neg) and ( len(pos) or activation=="linear" ):
            neg_str=" + ".join(str(x) for x in neg)
            print("    wire [%d:0] %s;" % ((swidth-2),sumname_neg))
            print("    assign %s = %s;" % (sumname_neg,neg_str))

        if len(pos) and len(neg):            
            print("    wire signed [%d:0] %s;" % ((swidth-1),sumname))
            print("    assign %s = $signed({1'b0,%s}) - $signed({1'b0,%s});" % (sumname,sumname_pos,sumname_neg))
        elif len(pos) and not len(neg):
            print()
            print("    //WARN: only positive weights. Using identity")
            print("    wire signed [%d:0] %s;" % ((swidth-1),sumname))
            print("    assign %s = $signed({1'b0,%s});" % (sumname, sumname_pos))
        elif not len(pos) and len(neg) and activation=="linear":
            print()
            print("    //WARN: only negative weights with linear. Negate.")
            print("    wire signed [%d:0] %s;" % ((swidth-1),sumname))
            print("    assign %s = -$signed({1'b0,%s});" % (sumname, sumname_neg))
        elif not len(pos) and len(neg) and activation=="relu":
            print()
            print("    //WARN: only negative weights with relu. Using zero")
            print("    wire signed [%d:0] %s;" % ((swidth-1),sumname))
            print("    assign %s = $signed({%d{1'b0}});" % (sumname,swidth))            
        elif not len(pos) and not len(neg):
            print()
            print("    //WARN: no weights. Using zero")
            print("    wire signed [%d:0] %s;" % ((swidth-1),sumname))
            print("    assign %s = $signed({%d{1'b0}});" % (sumname,swidth))
        print()
        
        #fix relu dimensions
        sumname0=sumname+"_f"
        if fixrwidth > 0:
            sumname='{'+sumname+','+str(fixrwidth)+"'b"+binary_repr(0,fixrwidth)+'}'
            swidth=fixrwidth+swidth
            decimal_s=decimal_r
        print("    wire signed [%d:0] %s;" % ((swidth-1),sumname0))
        print("    assign %s = $signed(%s);" % (sumname0, sumname))
        
        qrelu=prefix+"_qrelu"
        msb_sat=decimal_s+(rwidth-decimal_r)-1
        lsb_sat=msb_sat-rwidth+1
        
        if activation == "relu":
            print("    //relu")
            if rwidth >= swidth-1:
                if (lenMaxVal is not None):
                    print("    wire [%d:0] %s;" % ((lenMaxVal-1),prefix))
                    print("    assign %s = (%s<0) ? $unsigned({%d{1'b0}}) : $unsigned(%s);" % (prefix,sumname0,lenMaxVal, sumname0 ))
                else:
                    print("    wire [%d:0] %s;" % ((rwidth-1),prefix))
                    print("    assign %s = (%s<0) ? $unsigned({%d{1'b0}}) : $unsigned(%s);" % (prefix,sumname0,rwidth, sumname0 ))
            elif (sum(map(abs,weights))==0):
                print("    wire [%d:0] %s, %s;" % ((rwidth-1),prefix,qrelu))
                print("    assign %s = $unsigned({%d{1'b0}});" % (prefix,rwidth ))

            else:
                print("    wire [%d:0] %s, %s;" % ((rwidth-1),prefix,qrelu))
                if swidth-1==msb_sat:
                    print ("    assign %s = %s[%d:%d];" %(qrelu, sumname0, msb_sat, lsb_sat))
                #elif fixrwidth > 0:
                    #print("    wire [%d:0] %s;" % ((swidth-1),sumname0))
                    #print("    assign %s = %s;" % (sumname0, sumname))
                #    print("    DW01_satrnd #(%d, %d, %d) USR_%s ( .din(%s[%d:0]), .tc(1'b0), .rnd(1'b0), .ov(), .sat(1'b1), .dout(%s));" % ((swidth-1), msb_sat, lsb_sat, prefix, sumname0, (swidth-2), qrelu))
                else: 
                    print("    DW01_satrnd #(%d, %d, %d) USR_%s ( .din(%s[%d:0]), .tc(1'b0), .rnd(1'b0), .ov(), .sat(1'b1), .dout(%s));" % ((swidth-1), msb_sat, lsb_sat, prefix, sumname0, (swidth-2), qrelu))
                print("    assign %s = (%s<0) ? $unsigned({%d{1'b0}}) : $unsigned(%s);" % (prefix,sumname0,rwidth, qrelu ))
        elif activation == "linear":
             print("    //linear")
             print("    wire signed [%d:0] %s;" % ((rwidth-1),prefix))
             if rwidth >= swidth:  
                 print("    assign %s = %s;" % (prefix,sumname0))
             else:
                 print("    DW01_satrnd #(%d, %d, %d) USR_%s ( .din(%s), .tc(1'b1), .rnd(1'b0), .ov(), .sat(1'b1), .dout(%s));" % (swidth, msb_sat, lsb_sat, prefix, sumname0, prefix))
        return rwidth
    
    #Use "self.write_argmax_comb" without providing "order" or "masks" for the exact design results.
    def write_argmax_comb(self, prefix, act, vwidth, iwidth, signed, order=None, masks=None):
        lvl=0
        vallist=list(act)
        if (order is not None):
            vallist = []
            for i in range(0, len(order[0])):
                if (len(order[0][i]) == 1):
                    vallist.append([act[order[0][i][0]]])
                else:
                    vallist.append([act[order[0][i][0]], act[order[0][i][1]]])
        
        print("// argmax inp: " + ', '.join(act))
        
        idxlist=[str(iwidth)+"'b"+binary_repr(i,iwidth) for i in range(len(act))]
        if (order is not None):
            temp = []
            for i in range(0, len(order[0])):
                if (len(order[0][i]) == 1):
                    temp.append([idxlist[order[0][i][0]]])
                else:
                    temp.append([idxlist[order[0][i][0]], idxlist[order[0][i][1]]])
            idxlist = temp
        
        if (order is not None):
            while (len(vallist) > 1 or (len(vallist) == 1 and (len(vallist[0]) == 2))):
                newV=[]
                newI=[]
                comp=0
                print("    //comp level %d" % lvl)
                # for i in range(0,len(vallist)-1,2):
                for i in range(0, len(order[lvl])):
                    cmpname="cmp_"+str(lvl)+"_"+str(2*i)
                    vname=prefix+"_val_"+str(lvl)+"_"+str(2*i)
                    iname=prefix+"_idx_"+str(lvl)+"_"+str(2*i)
                    if (len(order[lvl][i]) == 1):
                        vname1=vallist[i][0]
                        iname1=idxlist[i][0]
                        print("    wire %s;" % cmpname)
                        if signed:
                            print("    wire signed [%d:0] %s;" % ((vwidth-1),vname))
                        else:
                            print("    wire [%d:0] %s;" % ((vwidth-1),vname))
                        print("    wire [%d:0] %s;" % ((iwidth-1),iname))
                        
                        #if there is a mask provided, apply the masks
                        print("    assign {%s} = 1;" % (cmpname))
                        
                        print("    assign {%s} =  %s;" % (vname,vname1))
                        print("    assign {%s} =  %s;" % (iname,iname1))
                        print()
                    else:
                        vname1=vallist[i][0]
                        vname2=vallist[i][1]
                        iname1=idxlist[i][0]
                        iname2=idxlist[i][1]
                        print("    wire %s;" % cmpname)
                        if signed:
                            print("    wire signed [%d:0] %s;" % ((vwidth-1),vname))
                        else:
                            print("    wire [%d:0] %s;" % ((vwidth-1),vname))
                        print("    wire [%d:0] %s;" % ((iwidth-1),iname))
                        
                        #if there is a mask provided, apply the masks
                        signal_name1 = vname1
                        signal_name2 = vname2
                        if (masks is not None and lvl < len(masks)):
                            mask = masks[lvl][i] #TODO: [lvl] must be removed for next level comparisons
                            signal_name1 = "{"
                            signal_name2 = "{"
                            for j in range(len(mask)):
                                if (mask[j] == "1"):
                                    signal_name1 = signal_name1 + vname1 + "[" + str(len(mask)-j-1) + "], "
                                    signal_name2 = signal_name2 + vname2 + "[" + str(len(mask)-j-1) + "], "
                            signal_name1 = signal_name1[0:-2] + "}"
                            signal_name2 = signal_name2[0:-2] + "}"
                            print("    //current mask = %s" % (mask))
                        
                        if (signal_name1 == "}"):
                            print("    assign {%s} = 1;" % (cmpname))
                        else:
                            print("    assign {%s} = ( %s >= %s );" % (cmpname, signal_name1, signal_name2))
                        
                        print("    assign {%s} = ( %s ) ? %s : %s;" % (vname,cmpname,vname1,vname2))
                        print("    assign {%s} = ( %s ) ? %s : %s;" % (iname,cmpname,iname1,iname2))
                        print()
                    newV.append(vname)
                    newI.append(iname)
                # if len(vallist) % 2 == 1:
                    # newV.append(vallist[-1])
                    # newI.append(idxlist[-1])
                lvl+=1
                vallist = list(newV)
                idxlist = list(newI)
                
                if (lvl >= len(order)):
                    break
                
                temp = []
                for i in range(0, len(order[lvl])):
                    if (len(order[lvl][i]) == 1):
                        temp.append([vallist[order[lvl][i][0]]])
                    else:
                        temp.append([vallist[order[lvl][i][0]], vallist[order[lvl][i][1]]])
                vallist = temp
                
                temp = []
                for i in range(0, len(order[lvl])):
                    if (len(order[lvl][i]) == 1):
                        temp.append([idxlist[order[lvl][i][0]]])
                    else:
                        temp.append([idxlist[order[lvl][i][0]], idxlist[order[lvl][i][1]]])
                idxlist = temp
                
                #if we have only the first level masks, we should make the masks = None for the next levels
                #but if we have multiple level masks, the below line should be removed
                # masks = None
        else:
            while len(vallist) > 1:
                newV=[]
                newI=[]
                comp=0
                print("    //comp level %d" % lvl)
                for i in range(0,len(vallist)-1,2):
                    cmpname="cmp_"+str(lvl)+"_"+str(i)
                    vname=prefix+"_val_"+str(lvl)+"_"+str(i)
                    iname=prefix+"_idx_"+str(lvl)+"_"+str(i)
                    vname1=vallist[i]
                    vname2=vallist[i+1]
                    iname1=idxlist[i]
                    iname2=idxlist[i+1]
                    print("    wire %s;" % cmpname)
                    if signed:
                        print("    wire signed [%d:0] %s;" % ((vwidth-1),vname))
                    else:
                        print("    wire [%d:0] %s;" % ((vwidth-1),vname))
                    print("    wire [%d:0] %s;" % ((iwidth-1),iname))
                    
                    #if there is a mask provided, apply the masks
                    signal_name1 = vname1
                    signal_name2 = vname2
                    if (masks is not None and lvl < len(masks)):
                        mask = masks[lvl][i//2] #TODO: [lvl] must be removed for next level comparisons
                        signal_name1 = "{"
                        signal_name2 = "{"
                        for j in range(len(mask)):
                            if (mask[j] == "1"):
                                signal_name1 = signal_name1 + vname1 + "[" + str(len(mask)-j-1) + "], "
                                signal_name2 = signal_name2 + vname2 + "[" + str(len(mask)-j-1) + "], "
                        signal_name1 = signal_name1[0:-2] + "}"
                        signal_name2 = signal_name2[0:-2] + "}"
                        print("    //current mask = %s" % (mask))
                    
                    if (signal_name1 == "}"):
                        print("    assign {%s} = 1;" % (cmpname))
                    else:
                        print("    assign {%s} = ( %s >= %s );" % (cmpname, signal_name1, signal_name2))
                    
                    print("    assign {%s} = ( %s ) ? %s : %s;" % (vname,cmpname,vname1,vname2))
                    print("    assign {%s} = ( %s ) ? %s : %s;" % (iname,cmpname,iname1,iname2))
                    print()
                    newV.append(vname)
                    newI.append(iname)
                if len(vallist) % 2 == 1:
                    newV.append(vallist[-1])
                    newI.append(idxlist[-1])
                lvl+=1
                vallist = list(newV)
                idxlist = list(newI)
                #if we have only the first level masks, we should make the masks = None for the next levels
                #but if we have multiple level masks, the below line should be removed
                # masks = None
        return idxlist[-1]
    
    #Use "self.write_verilog_comb" without providing "neuron_masks", "order" or "masks" for the exact design results.
    def write_verilog_comb(self, fileHandle, sum_relu_size, neuron_masks=None, order=None, arg_masks=None):
        INPNAME = "inp"
        OUTNAME = "out"
        stdoutbckp = sys.stdout
        sys.stdout = fileHandle
        width_a = self.QX[0][1]
        weights = self.coefs
        biases = self.intercept
        
        REGRESSOR=False
        inp_num=len(weights[0][0])
        
        width_o=int(len(weights[-1])).bit_length()

        print("//weights:", weights)
        print("//intercepts:", biases)
        
        print("module top ("+INPNAME+", "+OUTNAME+");")
        print("input ["+str(inp_num*width_a-1)+":"+str(0)+"] " + INPNAME +";")
        print("output ["+str(width_o-1)+":"+str(0)+"] " + OUTNAME +";")
        print()
        
        act_next=[]
        act_next_size=[]
        for i in range(inp_num):
            a=INPNAME+"["+str((i+1)*width_a-1)+":"+str(i*width_a)+"]"
            act_next.append(a)
        
        ver_relu_size=0
        for j in range(len(weights)):
            act=list(act_next)
            act_next=[]
            act_next_size=[]
            for i in range(len(weights[j])):
                print("// layer: %d - neuron: %d" % (j,i) )
                prefix = "n_"+str(j)+"_"
                nweights=weights[j][i]
                bias=biases[j][i]
                nsum_relu_size=sum_relu_size[j]
                merge_list= [-1] * len(weights[j][i])
                for k in range(len(weights[j][i])):
                    for ii in range(i):
                        if abs(weights[j][i][k]) == abs(weights[j][ii][k]):
                            merge_list[k]=ii
                            break
                lenMaxVal = self.no_write_verilog_comb(sum_relu_size, neuron_masks)
                if j == len(weights)-1:
                    activation = self.last_layer
                else:
                    activation = "relu"
                #SET EMPTY MERGE LIST FOR NOW
                merge_list=[-1] * len(weights[j][i])
                if neuron_masks == None:
                    nmasks=[]
                else:
                    nmasks=neuron_masks[j][i]
                temp = self.write_neuron_comb(prefix, j, i, nmasks, activation, act, merge_list, nsum_relu_size, lenMaxVal)
                ver_relu_size=max(ver_relu_size, temp)
                prefix=prefix+str(i)
                act_next.append(prefix)
                print()
        
        vw=max(sum_relu_size[len(weights)-1][1])
        if vw ==32:
           vw=ver_relu_size
        iw=width_o
        prefix="argmax"
        print("// argmax: %d classes, need %d bits" % (len(weights[-1]),iw) )
        if self.last_layer == "linear":
            signed=True
        else:
            signed=False
        out = self.write_argmax_comb(prefix, act_next, vw, iw, signed, order, arg_masks)
        print("    assign "+OUTNAME+" = " + out + ";")
        print()
        print("endmodule")
        sys.stdout=stdoutbckp


def indentLine(line, numofIndent):
    temp = ""
    for i in range(0, numofIndent):
        temp = temp + "\t"
    temp = temp + line
    return temp

def generate_tb_inputs(f, qmlp, X_test):
    temp = qmlp.convertInputs(X_test)
    np.savetxt(f, temp.astype(int), fmt='%d', delimiter=' ')

def generate_tb_outputs(f, y_test):
    np.savetxt(f, y_test.astype(int), fmt='%d', delimiter='\n')

def generate_tb_comb(f, qmlp, period=200000000): #200ms by default
    stdoutbckp = sys.stdout
    sys.stdout = f
    
    print("`timescale 1ns/1ps")
    print("`define EOF 32'hFFFF_FFFF")
    print("`define NULL 0")
    print()
    print("module top_tb();")
    print()
    
    weight_list = qmlp.coefs
    width_o=get_width(len(weight_list[-1]))
    print(f"    parameter OUTWIDTH={str(width_o)};")
    print("    parameter NUM_A=" + str(len(qmlp.coefs[0][0])) + ";")
    print("    parameter WIDTH_A=" + str(qmlp.QX[0][1]) + ";")
    print()
    
    print("localparam period = " + str(period), end="")
    print('''.00;

    reg  [WIDTH_A-1:0] at[NUM_A-1:0];
    wire [NUM_A*WIDTH_A-1:0] inp;
    wire [OUTWIDTH-1:0] out;

    wire [WIDTH_A:0] r;

    top DUT(.inp(inp),
            .out(out)
            );


    integer inFile,outFile,i;
    initial
    begin
        $display($time, " << Starting the Simulation >>");
            inFile = $fopen("./sim/sim.Xtest","r");
        if (inFile == `NULL) begin
                $display($time, " file not found");
                $finish;
        end
        outFile = $fopen("./sim/output.txt");
        while (!$feof(inFile)) begin
            for (i=0;i<NUM_A;i=i+1) begin
                $fscanf(inFile,"%d ",at[i]);
            end
            $fscanf(inFile,"\\n");
            #(period)
            $fwrite(outFile,"%d\\n", out);
        end
        #(period)
        $display($time, " << Finishing the Simulation >>");
        $fclose(outFile);
        $fclose(inFile);
        $finish;
    end


    genvar gi;
    generate
    for (gi=0;gi<NUM_A;gi=gi+1) begin : genbit
        assign inp[(gi+1)*WIDTH_A-1:gi*WIDTH_A] = at[gi];
    end
    endgenerate


    endmodule''')
    sys.stdout = stdoutbckp

def generate_qrelu_rtl(f):
    file = open("./IP_Designs/DW01_satrnd.v", "r")
    temp = file.read()
    f.write(temp)
    file.close()

def get_width (a):
    return int(a).bit_length()

SHIFT_OR_CONCAT=1
def get_ones (n):
    if SHIFT_OR_CONCAT:
        return [i for i in range(0, n.bit_length()) if (n >> i & 1)]
    else:
        return [n >> i & 1 for i in range(0, n.bit_length())]
