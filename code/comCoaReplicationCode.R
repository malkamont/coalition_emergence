#############
## PREPARE ##
#############

#workspace
library(igraph)
library(pewmethods)

##########################
# COLLABORATION PREPARIS #
##########################

#collaboration
a = read.table("/m/triton/scratch/work/malkama5/complexCoalition/networkData/attributesRoster.txt", header = TRUE, sep = ";")
d = read.table("/m/triton/scratch/work/malkama5/complexCoalition/networkData/collaboration2014.txt", header = TRUE, sep = ";")
d = subset(d, to %in% a$actor[a$roster2014 == "yes"] & from != "FI061" & from != "FI090")
g = simplify(graph_from_data_frame(d, directed = TRUE), remove.loops = TRUE, remove.multiple = TRUE)

#non-respondents
i = degree(g, mode = "out")
i = names(i[i == 0])

#beliefs
b = read.table("/m/triton/scratch/work/malkama5/complexCoalition/networkData/beliefs2014.txt", header = TRUE, sep = ";")
b = subset(b, status != "non-respondent" & status != "invalid")
rownames(b) = b$actor
b = b[c(3:9, 13:22, 26:34)]
mis = (sum(is.na(b))) / (nrow(b) * ncol(b))

#remove respondents with less than five valid responses
bm = b
bm[bm > 1] = 1
bm = rowSums(bm, na.rm = TRUE)
bm = names(bm[bm < 5])
b = subset(b, !(rownames(b) %in% bm))
fis = (sum(is.na(b))) / (nrow(b) * ncol(b))

#impute remaining missing responses
s = b
s[s > 0 | is.na(s)] = 0
r = 10
for (seed in 1:r){
s = s + impute_vars(b, to_impute = names(b), method = "ranger", seed = seed)}
b = round(s / r, 1)

#average over responses of alters for respondents with missing responses
j = i[! i %in% rownames(b)]
for (v in j){
rc = subset(d, to == v)
rc = subset(rc, !(from %in% i))
rc = subset(b, rownames(b) %in% rc$from)
rc = round(colSums(rc) / nrow(rc), 1)
rc = data.frame(t(data.frame(rc)))
rownames(rc) = v
b = rbind(b, rc)}

#distance
ds = psych::describe(b)
ds = ds[order(ds$sd, decreasing = TRUE),]
b = b[rownames(ds)[ds$sd > 1]]
b = as.matrix(dist(b, method = "manhattan"))
b = (max(b) - b) / max(b)
b = 1 - b

#intersect
g = as.undirected(g, mode = "each")
E(g)$weight = 1
g = simplify(g, edge.attr.comb = "sum")
m = as_adjacency_matrix(g, attr = "weight", sparse = FALSE)
m[m == 2] = 1
m = m[rownames(m), colnames(m)] + b[rownames(m), colnames(m)] - 1
m[m < 0] = 0

#graph
g = graph_from_adjacency_matrix(m, mode = "upper", weighted = TRUE, diag = FALSE)
E(g)$weight = round(scales::rescale(E(g)$weight, to = c(1, 10000)), 0)
am = as_data_frame(g)
colnames(am) = c("src", "trg", "weight")
am$layer = 0
c0 = am

###########################
# COLLABORATION POSTPARIS #
###########################

#collaboration
a = read.table("/m/triton/scratch/work/malkama5/complexCoalition/networkData/attributesRoster.txt", header = TRUE, sep = ";")
d = read.table("/m/triton/scratch/work/malkama5/complexCoalition/networkData/collaboration2020.txt", header = TRUE, sep = ";")
d = subset(d, to %in% a$actor[a$roster2020 == "yes"] & from != "FI061" & from != "FI090")
g = simplify(graph_from_data_frame(d, directed = TRUE), remove.loops = TRUE, remove.multiple = TRUE)

#non-respondents
i = degree(g, mode = "out")
i = names(i[i == 0])

#beliefs
b = read.table("/m/triton/scratch/work/malkama5/complexCoalition/networkData/beliefs2020.txt", header = TRUE, sep = ";")
b = subset(b, status != "non-respondent" & status != "invalid")
rownames(b) = b$actor
b = b[3:34]
mis = (sum(is.na(b))) / (nrow(b) * ncol(b))

#remove respondents with less than five valid responses
bm = b
bm[bm > 1] = 1
bm = rowSums(bm, na.rm = TRUE)
bm = names(bm[bm < 5])
b = subset(b, !(rownames(b) %in% bm))
fis = (sum(is.na(b))) / (nrow(b) * ncol(b))

#impute remaining missing responses
s = b
s[s > 0 | is.na(s)] = 0
r = 10
for (seed in 1:r){
s = s + impute_vars(b, to_impute = names(b), method = "ranger", seed = seed)}
b = round(s / r, 1)

#average over responses of alters for respondents with missing responses
j = i[! i %in% rownames(b)]
for (v in j){
rc = subset(d, to == v)
rc = subset(rc, !(from %in% i))
rc = subset(b, rownames(b) %in% rc$from)
rc = round(colSums(rc) / nrow(rc), 1)
rc = data.frame(t(data.frame(rc)))
rownames(rc) = v
b = rbind(b, rc)}

#distance
ds = psych::describe(b)
ds = ds[order(ds$sd, decreasing = TRUE),]
b = b[rownames(ds)[ds$sd > 1]]
b = as.matrix(dist(b, method = "manhattan"))
b = (max(b) - b) / max(b)
b = 1 - b

#intersect
g = as.undirected(g, mode = "each")
E(g)$weight = 1
g = simplify(g, edge.attr.comb = "sum")
m = as_adjacency_matrix(g, attr = "weight", sparse = FALSE)
m[m == 2] = 1
m = m[rownames(m), colnames(m)] + b[rownames(m), colnames(m)] - 1
m[m < 0] = 0

#graph
g = graph_from_adjacency_matrix(m, mode = "upper", weighted = TRUE, diag = FALSE)
E(g)$weight = round(scales::rescale(E(g)$weight, to = c(1, 10000)), 0)
am = as_data_frame(g)
colnames(am) = c("src", "trg", "weight")
am$layer = 1
c1 = am

######################
# DISCOURSE PREPARIS #
######################

#discourse
d = as.matrix(read.table("/m/triton/scratch/work/malkama5/complexCoalition/networkData/discourse0.txt", sep = ";", header = TRUE, row.names = 1))
d = d %*% t(d)
g = graph_from_adjacency_matrix(d, mode = "upper", weighted = TRUE, diag = FALSE)

#average activity
adj = as_adjacency_matrix(g, sparse = FALSE, attr = "weight")
diag(adj) = merge(data.frame("vertex" = V(g)$name), data.frame("vertex" = rownames(d), "diag" = diag(d)), by = "vertex", sort = FALSE, all.x = TRUE)$diag
aa = adj
ac = diag(adj)

for (row in rownames(aa)){
    for (col in colnames(aa)){
        aa[row, col] = mean(ac[c(row, col)])}}
d = adj / aa

g = graph_from_adjacency_matrix(d, mode = "upper", weighted = TRUE, diag = FALSE)
E(g)$weight = round(scales::rescale(E(g)$weight, to = c(1, 10000)), 0)
am = as_data_frame(g)
colnames(am) = c("src", "trg", "weight")
am$layer = 2
d0 = am

#######################
# DISCOURSE POSTPARIS #
#######################

#discourse
d = as.matrix(read.table("/m/triton/scratch/work/malkama5/complexCoalition/networkData/discourse1.txt", sep = ";", header = TRUE, row.names = 1))
d = d %*% t(d)
g = graph_from_adjacency_matrix(d, mode = "upper", weighted = TRUE, diag = FALSE)

#average activity
adj = as_adjacency_matrix(g, sparse = FALSE, attr = "weight")
diag(adj) = merge(data.frame("vertex" = V(g)$name), data.frame("vertex" = rownames(d), "diag" = diag(d)), by = "vertex", sort = FALSE, all.x = TRUE)$diag
aa = adj
ac = diag(adj)

for (row in rownames(aa)){
    for (col in colnames(aa)){
        aa[row, col] = mean(ac[c(row, col)])}}
d = adj / aa

g = graph_from_adjacency_matrix(d, mode = "upper", weighted = TRUE, diag = FALSE)
E(g)$weight = round(scales::rescale(E(g)$weight, to = c(1, 10000)), 0)
am = as_data_frame(g)
colnames(am) = c("src", "trg", "weight")
am$layer = 3
d1 = am

##########################
# COMMUNICATION PREPARIS #
##########################

#communication
#a = read.table("/m/triton/scratch/work/malkama5/complexCoalition/networkData/attributesRoster.txt", header = TRUE, sep = ";")
#k = paste(scan("/m/triton/scratch/work/malkama5/complexCoalition/networkData/twitterKeys.txt", character(), encoding = "UTF-8"), collapse = "|")
#d = readRDS("/m/triton/scratch/work/malkama5/complexCoalition/networkData/twitterFinland2013_16.rds")
#d = subset(d, !is.na(ref_org))
#d = subset(d, type == "retweeted" & level != "3" & ref_level != "3" & created_at >= as.Date("2013-01-01") & created_at <= as.Date("2016-12-31"))
#d$loop = ifelse(d$org == d$ref_org, TRUE, FALSE)
#d$valid = stringi::stri_count_regex(str = d$text, pattern = k, case_insensitive = TRUE)
#d = subset(d, loop == FALSE & valid > 0)

#g = graph_from_data_frame(d[c(which(names(d) == "ref_org"), which(names(d) == "org"))], directed = FALSE)
#E(g)$weight = 1
#g = simplify(g, edge.attr.comb = "sum")
#drop = a$actor[a$roster2014 == "no"]
#for (i in 1:length(drop)){
#g = delete_vertices(g, V(g)[V(g)$name == drop[i]])}
#write_graph(g, "/m/triton/scratch/work/malkama5/complexCoalition/networkData/communication0.xml", format = "graphml")
g = read_graph("/m/triton/scratch/work/malkama5/complexCoalition/networkData/communication0.xml", format = "graphml")

#average activity
adj = as_adjacency_matrix(g, sparse = FALSE, attr = "weight")
diag(adj) = rowSums(adj)
aa = adj
ac = diag(adj)

for (row in rownames(aa)){
    for (col in colnames(aa)){
        aa[row, col] = mean(ac[c(row, col)])}}
d = adj / aa

g = graph_from_adjacency_matrix(d, mode = "upper", weighted = TRUE, diag = FALSE)
E(g)$weight = round(scales::rescale(E(g)$weight, to = c(1, 10000)), 0)
#E(g)$weight = round(scales::rescale(log(1 + E(g)$weight), to = c(1, 10000)), 0)
am = as_data_frame(g)
colnames(am) = c("src", "trg", "weight")
am$layer = 4
o0 = am

###########################
# COMMUNICATION POSTPARIS #
###########################

#communication
#a = read.table("/m/triton/scratch/work/malkama5/complexCoalition/networkData/attributesRoster.txt", header = TRUE, sep = ";")
#k = paste(scan("/m/triton/scratch/work/malkama5/complexCoalition/networkData/twitterKeys.txt", character(), encoding = "UTF-8"), collapse = "|")
#d = readRDS("/m/triton/scratch/work/malkama5/complexCoalition/networkData/twitterFinland2017_21.rds")
#d = subset(d, !is.na(ref_org))
#d = subset(d, type == "retweeted" & level != "3" & ref_level != "3" & created_at >= as.Date("2017-01-01") & created_at <= as.Date("2020-12-31"))
#d$loop = ifelse(d$org == d$ref_org, TRUE, FALSE)
#d$valid = stringi::stri_count_regex(str = d$text, pattern = k, case_insensitive = TRUE)
#d = subset(d, loop == FALSE & valid > 0)

#g = graph_from_data_frame(d[c(which(names(d) == "ref_org"), which(names(d) == "org"))], directed = FALSE)
#E(g)$weight = 1
#g = simplify(g, edge.attr.comb = "sum")
#drop = a$actor[a$roster2020 == "no"]
#for (i in 1:length(drop)){
#g = delete_vertices(g, V(g)[V(g)$name == drop[i]])}
#write_graph(g, "/m/triton/scratch/work/malkama5/complexCoalition/networkData/communication1.xml", format = "graphml")
g = read_graph("/m/triton/scratch/work/malkama5/complexCoalition/networkData/communication1.xml", format = "graphml")

#average activity
adj = as_adjacency_matrix(g, sparse = FALSE, attr = "weight")
diag(adj) = rowSums(adj)
aa = adj
ac = diag(adj)

for (row in rownames(aa)){
    for (col in colnames(aa)){
        aa[row, col] = mean(ac[c(row, col)])}}
d = adj / aa

g = graph_from_adjacency_matrix(d, mode = "upper", weighted = TRUE, diag = FALSE)
E(g)$weight = round(scales::rescale(E(g)$weight, to = c(1, 10000)), 0)
#E(g)$weight = round(scales::rescale(log(1 + E(g)$weight), to = c(1, 10000)), 0)
am = as_data_frame(g)
colnames(am) = c("src", "trg", "weight")
am$layer = 5
o1 = am

###########
## STORE ##
###########

#system
fs = rbind(c0, c1, d0, d1, o0, o1)
write.table(fs, "/m/triton/scratch/work/malkama5/complexCoalition/networkData/fs.txt", quote = TRUE, row.names = FALSE, col.names = FALSE, sep = ";")

#####################
### VISUALISATION ###
#####################

#layer similarity
library(igraph)
library(multinet)
bb = read.table("/m/triton/scratch/work/malkama5/complexCoalition/networkData/bb.txt", sep = ";", header = TRUE)
m = matrix(1, length(unique(bb$layer)), length(unique(bb$layer)))
n = c("Res/T0", "Res/T1", "Dis/T0", "Dis/T1", "Com/T0", "Com/T1")
dimnames(m) = list(n, n)
M = m

for (k in unique(bb$layer)){
    for (l in unique(bb$layer)){
        x = graph_from_data_frame(subset(bb, layer == k), directed = FALSE)
        x = as_adjacency_matrix(x, sparse = FALSE)
        y = graph_from_data_frame(subset(bb, layer == l), directed = FALSE)
        y = as_adjacency_matrix(y, sparse = FALSE)
        o = table(c(rownames(x), rownames(y)))
        o = names(o[o == 2])
        #d = list(x[o, o], y[o, o])
        #m[k + 1, l + 1] = round(1 - NetworkDistance::nd.dsd(d, out.dist = FALSE, type = "NLap")$D[1, 2], 2)}}
        x = graph_from_adjacency_matrix(x[o, o], mode = "undirected", diag = FALSE)
        y = graph_from_adjacency_matrix(y[o, o], mode = "undirected", diag = FALSE)
        xy = ml_empty()
        add_igraph_layer_ml(xy, x, "x")
        add_igraph_layer_ml(xy, y, "y")
        m[k + 1, l + 1] = round(layer_comparison_ml(xy, method = "jaccard.edges"), 2)[1, 2]
        M[k + 1, l + 1] = round(cor(degree(x), degree(y), method = "kendall"), 2)}}

#layer degree entropy
de = list()
for (l in unique(bb$layer)){
    x = graph_from_data_frame(subset(bb, layer == l), directed = FALSE)
    ml = ml_empty()
    add_igraph_layer_ml(ml, x, "x")
    de[[l + 1]] = round(layer_summary_ml(ml, "x", method = "entropy.degree"), 2)}

#visualise similarity
library(ggplot2)
cpie = c("#008e97", "#fc4c02", "#C204af", "#49d5de", "#d9774e", "#C771be")

m[lower.tri(m, diag = TRUE)] = NA #jaccard
m = m[c(1:5), c(2:6)]
mn = round(min(m, na.rm = TRUE), 2)
mx = round(max(m, na.rm = TRUE), 2)
m = as.data.frame(as.table(as.matrix(m)))
m$Var1 = factor(m$Var1, ordered = TRUE, levels = levels(m$Var1)[length(levels(m$Var1)):1])
m$Lab = format(m$Freq, nsmall = 2)

p0 = ggplot(m, aes(x = Var2, y = Var1, fill = Freq)) +
geom_tile(color = "black") +
geom_text(aes(label = Lab), size = 3, color = "black", family = "Times") +
scale_fill_gradient(low = cpie[1], high = cpie[2], na.value = "black", limits = c(mn, mx), breaks = c(mn, mx)) +
theme(legend.position = "bottom", legend.title = element_blank(), legend.key.height = unit(0.3, "cm"), legend.key.width = unit(0.7, "cm"), plot.title = element_text(size = 8, hjust = 0.5), axis.ticks = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.border = element_blank(), panel.background = element_blank(), text = element_text(family = "Times", color = "black"), axis.text.x = element_text(size = 8, color = "black", angle = 45, vjust = 1, hjust = 1), axis.text.y = element_text(size = 8, color = "black")) +
labs(title = "Jaccard similarity for edges", x = NULL, y = NULL) +
guides(fill = guide_colorbar(frame.colour = "white", ticks = FALSE))

M[lower.tri(M, diag = TRUE)] = NA #kendall
M = M[c(1:5), c(2:6)]
mn = round(min(M, na.rm = TRUE), 2)
mx = round(max(M, na.rm = TRUE), 2)
M = as.data.frame(as.table(as.matrix(M)))
M$Var1 = factor(M$Var1, ordered = TRUE, levels = levels(M$Var1)[length(levels(M$Var1)):1])
M$Lab = format(M$Freq, nsmall = 2)

p1 = ggplot(M, aes(x = Var2, y = Var1, fill = Freq)) +
geom_tile(color = "black") +
geom_text(aes(label = Lab), size = 3, color = "black", family = "Times") +
scale_fill_gradient(low = cpie[1], high = cpie[2], na.value = "black", limits = c(mn, mx), breaks = c(mn, mx)) +
theme(legend.position = "bottom", legend.title = element_blank(), legend.key.height = unit(0.3, "cm"), legend.key.width = unit(0.7, "cm"), plot.title = element_text(size = 8, hjust = 0.5), axis.ticks = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.border = element_blank(), panel.background = element_blank(), text = element_text(family = "Times", color = "black"), axis.text.x = element_text(size = 8, color = "black", angle = 45, vjust = 1, hjust = 1), axis.text.y = element_text(size = 8, color = "white")) +
labs(title = "Kendall correlation for degrees", x = NULL, y = NULL) +
guides(fill = guide_colorbar(frame.colour = "white", ticks = FALSE))

ge = ggpubr::ggarrange(p0, p1, nrow = 1, ncol = 2)
ggsave(plot = ge, filename = "/m/triton/scratch/work/malkama5/complexCoalition/networkPlot/comCoaFig_x_comparison.png", width = 5.5, height = 3.5, dpi = 1000)

#degree ridge
library(igraph)
library(ggplot2)
library(ggridges)
cpie = c("#008e97", "#fc4c02", "#C204af", "#49d5de", "#d9774e", "#C771be")

ds = read.table("/m/triton/scratch/work/malkama5/complexCoalition/networkData/bb.txt", sep = ";", header = TRUE)
df = data.frame(actor = unique(c(ds$source, ds$target)))
n = c("Res/T0", "Res/T1", "Dis/T0", "Dis/T1", "Com/T0", "Com/T1")
for (l in unique(ds$layer)){
    g = graph_from_data_frame(subset(ds, layer == l), directed = FALSE)
    dg = data.frame(actor = V(g)$name, dg = scale(degree(g)))
    names(dg) = c("actor", n[l + 1])
    df = merge(x = df, y = dg, by = "actor", all.x = TRUE)}

rownames(df) = df$actor
df = df[2:ncol(df)]
df = as.data.frame(as.table(as.matrix(df)))
df$period[df$Var2 == "Res/T0" | df$Var2 == "Dis/T0" | df$Var2 == "Com/T0"] = "T0"
df$period[df$Var2 == "Res/T1" | df$Var2 == "Dis/T1" | df$Var2 == "Com/T1"] = "T1"

p = ggplot(na.omit(df), aes(x = Freq, y = Var2)) +
geom_density_ridges(aes(fill = period, color = period), size = 0.0) +
scale_fill_manual(values = alpha(c(cpie[1], cpie[2]), 0.5)) +
scale_color_manual(values = c(cpie[1], cpie[2])) +
theme(legend.position = "right", legend.title = element_blank(), legend.key.height = unit(0.1, "cm"), legend.key.width = unit(0.7, "cm"), legend.text = element_text(size = 8), plot.title = element_text(size = 8, hjust = 0.5), axis.ticks = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.border = element_blank(), panel.background = element_blank(), text = element_text(family = "Times", color = "black"), axis.text.x = element_text(size = 8, color = "black", angle = 0, vjust = 1, hjust = 1), axis.text.y = element_text(size = 8, color = "black"), axis.title.y = element_text(size = 8)) +
labs(title = "Degree distribution", x = NULL, y = "Percentage") +
xlim(-4, 4)
ggsave(plot = p, filename = "/m/triton/scratch/work/malkama5/complexCoalition/networkPlot/comCoaFig_x_degree.png", width = 5, height = 6, dpi = 1000)

#mutual information
library(ggplot2)
cpie = c("#008e97", "#fc4c02", "#C204af", "#49d5de", "#d9774e", "#C771be")

m = read.table("/m/triton/scratch/work/malkama5/complexCoalition/networkResult/mi_2595.txt", sep = ";", header = FALSE) #monolayer
n = c("Res/T0", "Res/T1", "Dis/T0", "Dis/T1", "Com/T0", "Com/T1")
dimnames(m) = list(n, n)

m[lower.tri(m, diag = TRUE)] = NA
m = m[c(1:5), c(2:6)]
mn0 = round(min(m, na.rm = TRUE), 2)
mx0 = round(max(m, na.rm = TRUE), 2)
m = as.data.frame(as.table(as.matrix(m)))
m$Var1 = factor(m$Var1, ordered = TRUE, levels = levels(m$Var1)[length(levels(m$Var1)):1])
m$Lab = format(m$Freq, nsmall = 2)

M = read.table("/m/triton/scratch/work/malkama5/complexCoalition/networkResult/mi_2681.txt", sep = ";", header = FALSE) #multilayer
n = c("Res/T0", "Res/T1", "Dis/T0", "Dis/T1", "Com/T0", "Com/T1")
dimnames(M) = list(n, n)

M[lower.tri(M, diag = TRUE)] = NA
M = M[c(1:5), c(2:6)]
mn1 = round(min(M, na.rm = TRUE), 2)
mx1 = round(max(M, na.rm = TRUE), 2)
M = as.data.frame(as.table(as.matrix(M)))
M$Var1 = factor(M$Var1, ordered = TRUE, levels = levels(M$Var1)[length(levels(M$Var1)):1])
M$Lab = format(M$Freq, nsmall = 2)

mn = min(c(mn0, mn1))
mx = max(c(mx0, mx1))

p0 = ggplot(m, aes(x = Var2, y = Var1, fill = Freq)) +
geom_tile(color = "black") +
geom_text(aes(label = Lab), size = 3, color = "black", family = "Times") +
scale_fill_gradient(low = cpie[1], high = cpie[2], na.value = "black", limits = c(mn, mx), breaks = c(mn, mx)) +
theme(legend.text = element_text(color = "black"), legend.position = "bottom", legend.title = element_blank(), legend.key.height = unit(0.3, "cm"), legend.key.width = unit(0.7, "cm"), plot.title = element_text(size = 8, hjust = 0.5), axis.ticks = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.border = element_blank(), panel.background = element_blank(), text = element_text(family = "Times", color = "black"), axis.text.x = element_text(size = 8, color = "black", angle = 45, vjust = 1, hjust = 1), axis.text.y = element_text(size = 8, color = "black")) +
labs(title = "Monolayer", x = NULL, y = NULL) +
guides(fill = guide_colorbar(frame.colour = "white", ticks = FALSE))

p1 = ggplot(M, aes(x = Var2, y = Var1, fill = Freq)) +
geom_tile(color = "black") +
geom_text(aes(label = Lab), size = 3, color = "black", family = "Times") +
scale_fill_gradient(low = cpie[1], high = cpie[2], na.value = "black", limits = c(mn, mx), breaks = c(mn, mx)) +
theme(legend.text = element_text(color = "black"), legend.position = "bottom", legend.title = element_blank(), legend.key.height = unit(0.3, "cm"), legend.key.width = unit(0.7, "cm"), plot.title = element_text(size = 8, hjust = 0.5), axis.ticks = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.border = element_blank(), panel.background = element_blank(), text = element_text(family = "Times", color = "black"), axis.text.x = element_text(size = 8, color = "black", angle = 45, vjust = 1, hjust = 1), axis.text.y = element_text(size = 8, color = "white")) +
labs(title = "Multilayer", x = NULL, y = NULL) +
guides(fill = guide_colorbar(frame.colour = "white", ticks = FALSE))

ge = ggpubr::ggarrange(p0, p1, nrow = 1, ncol = 2)
ggsave(plot = ge, filename = "/m/triton/scratch/work/malkama5/complexCoalition/networkPlot/comCoaFig_x_mutual.png", width = 5.5, height = 3.5, dpi = 1000)

#community heatmap
library(ggplot2)
cpie = c("#008e97", "#fc4c02", "#C204af", "#49d5de", "#d9774e", "#C771be")

at = read.table("/m/triton/scratch/work/malkama5/complexCoalition/networkData/attributesRoster.txt", header = TRUE, sep = ";")
dl = read.table("/m/triton/scratch/work/malkama5/complexCoalition/networkData/dl_2681.txt", sep = ";", header = TRUE, col.names = c("actor", "layer", "member"))
hm = data.frame(actor = unique(c(dl$actor, at$actor)))
hm = merge(x = hm, y = at, by = "actor", all.x = TRUE)
hm$type = gsub("_", " ", stringr::str_to_title(hm$type))
hm$type[hm$type == "Non governmental"] = "Non-governmental"
for (l in 1:length(unique(dl$layer))){
    z = subset(dl, layer == l - 1)[c(1, 3)]
    names(z)[2] = l - 1
    hm = merge(x = hm, y = z, by = "actor", all.x = TRUE)}
hm = hm[order(hm$type, hm$name),][c(2:5, 8:13)]
hm$type = paste(hm$type, 1:nrow(hm))
names(hm) = c("name", "type", "roster2014", "roster2020", "Res/T0", "Res/T1", "Dis/T0", "Dis/T1", "Com/T0", "Com/T1")
hm = hm[order(hm$"Res/T1", hm$"Dis/T1", hm$"Com/T1", hm$"Res/T0", hm$"Dis/T0", hm$"Com/T0"),]
m = max(hm[5:ncol(hm)], na.rm = TRUE) + 1
hm$"Res/T0"[hm$roster2014 == "no"] = m
hm$"Dis/T0"[hm$roster2014 == "no"] = m
hm$"Com/T0"[hm$roster2014 == "no"] = m
hm$"Res/T1"[hm$roster2020 == "no"] = m
hm$"Dis/T1"[hm$roster2020 == "no"] = m
hm$"Com/T1"[hm$roster2020 == "no"] = m
hm = reshape2::melt(hm[-c(3:4)], id.vars = c("name", "type"))
hm$variable = as.character(hm$variable)
hm$variable = factor(hm$variable, levels = c("Com/T1", "Dis/T1", "Res/T1", "Com/T0", "Dis/T0", "Res/T0"))
hm$row = 1:nrow(hm)

h = ggplot(hm, aes(reorder(name, row), variable, fill = factor(value))) +
geom_tile(color = "black") +
scale_fill_manual(values = c(cpie[1:3], "lightgrey"), na.value = "white") +
labs(title = NULL, x = NULL, y = NULL) +
theme(text = element_text(color = "black", family = "Times"), axis.text.x = element_text(size = 3, color = "black", family = "Times", angle = 45, vjust = 1, hjust = 1), axis.text.y = element_text(size = 10, color = "black", family = "Times"), legend.position = "none", axis.ticks = element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.border = element_blank(), panel.background = element_blank(), plot.title = element_text(hjust = 0.5))
ggsave(plot = h, filename = "/m/triton/scratch/work/malkama5/complexCoalition/networkPlot/comCoaFig_x_heatmap.png", width = 10, height = 6, dpi = 1000)

#compare assignments
options(repr.matrix.max.rows = 200, repr.matrix.max.cols = 10)
library(ggplot2)
at = read.table("/m/triton/scratch/work/malkama5/complexCoalition/networkData/attributesRoster.txt", header = TRUE, sep = ";")

mo = read.table("/m/triton/scratch/work/malkama5/complexCoalition/networkData/dl_2595.txt", sep = ";", header = TRUE, col.names = c("actor", "layer", "member_mo"))
mu = read.table("/m/triton/scratch/work/malkama5/complexCoalition/networkData/dl_2681.txt", sep = ";", header = TRUE, col.names = c("actor", "layer", "member_mu"))
hm = cbind(mo, mu[3])
hm = merge(x = hm, y = at[1:3], by = "actor", all.x = TRUE, sort = FALSE)

sl = subset(hm, layer == 1)
sl[order(sl$member_mo, sl$member_mo),]

#climate indicators
library(ggplot2)
idc = read.table("/m/triton/scratch/work/malkama5/complexCoalition/networkData/climateIndicators.txt", sep = ";", header = TRUE)
rownames(idc) = idc$row
idc = idc[9:16, 15:ncol(idc)]
names(idc) = substring(names(idc), 2, 5)
idc = as.data.frame(as.table(as.matrix(idc)))
idc = na.omit(idc)

cpie = c("#008e97", "#49d5de", "#fc4c02", "#d9774e", "#c204af", "#c771be", "#39ff14", "#6fba61")
idc$colo[idc$Var1 == "Carbon emission"] = cpie[1]
idc$colo[idc$Var1 == "Carbon sink"] = cpie[2]
idc$colo[idc$Var1 == "Net emission"] = cpie[3]
idc$colo[idc$Var1 == "Consumption"] = cpie[4]
idc$colo[idc$Var1 == "Total energy"] = cpie[5]
idc$colo[idc$Var1 == "Clean energy"] = cpie[6]
idc$colo[idc$Var1 == "Wealth inequality"] = cpie[7]
idc$colo[idc$Var1 == "Gross product"] = cpie[8]

idg = ggplot(idc, aes(x = Var2, y = Freq, group = factor(Var1), color = factor(Var1))) +
geom_line(size = 0.5, lineend = "round", linejoin = "bevel") +
geom_point(size = 1) +
scale_color_manual(values = cpie) +
labs(x = NULL, y = NULL) +
ylim(-1.3, 1.3) +
theme(text = element_text(color = "black", family = "Times"), axis.text.x = element_text(color = "black", family = "Times", angle = 30, vjust = 1, hjust = 1), axis.text.y = element_text(color = "black", family = "Times"), axis.ticks = element_blank(), legend.title = element_blank(), legend.key = element_rect(fill = NA), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.border = element_blank(), panel.background = element_blank())
ggsave(plot = idg, filename = "/m/triton/scratch/work/malkama5/complexCoalition/networkPlot/comCoaFig_x_indicator.png", width = 10, height = 6, dpi = 1000)
