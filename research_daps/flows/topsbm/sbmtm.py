import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import graph_tool.all as gt
import numpy as np
import pandas as pd


class sbmtm:
    """Topic-modeling with hierarchical Stochastic Block Models.

    Attributes:
        g: Bipartite graphtool network of documents and words
        word: List of unique words (word-nodes)
        documents: List of document titles (doc-nodes)
        state: Inference state from graphtool
        groups: Results of topic and cluster membership
        mdl: Minimum description legnth of inferred state
        L: Number of levels in hierarchy
    """

    g: gt.Graph
    words: List[str]
    documents: List[str]
    state: gt.inference.nested_blockmodel.NestedBlockState
    groups: Dict[int, Dict[str, Union[int, np.ndarray]]]
    mdl: float
    L: int

    def __init__(self):
        self.g = None  # network

        self.words = []  # list of word nodes
        self.documents = []  # list of document nodes

        self.state = None  # inference state from graphtool
        self.groups = {}  # results of group membership from inference
        self.mdl = np.nan  # minimum description length of inferred state
        self.L = None  # number of levels in hierarchy

    def make_graph(
        self,
        list_texts: List[str],
        documents: Optional[List[str]] = None,
        counts=True,
        n_min=None,
    ) -> None:
        """Load a corpus and generate the word-document network.

        Args:
            list_texts: List of documents
            documents: Titles of documents
            counts: Save edge-multiplicity as counts
            n_min: Filter all word-nodes with less than n_min counts
        """
        D = len(list_texts)

        ## if there are no document titles, we assign integers 0,...,D-1
        ## otherwise we use supplied titles
        if documents is None:
            list_titles = [str(h) for h in range(D)]
        else:
            list_titles = documents

        ## make a graph
        ## create a graph
        g = gt.Graph(directed=False)
        ## define node properties
        ## name: docs - title, words - 'word'
        ## kind: docs - 0, words - 1
        name = g.vp["name"] = g.new_vp("string")
        kind = g.vp["kind"] = g.new_vp("int")
        if counts:
            ecount = g.ep["count"] = g.new_ep("int")

        docs_add: dict = defaultdict(lambda: g.add_vertex())
        words_add: dict = defaultdict(lambda: g.add_vertex())

        ## add all documents first
        for i_d in range(D):
            title = list_titles[i_d]
            d = docs_add[title]

        ## add all documents and words as nodes
        ## add all tokens as links
        for i_d in range(D):
            title = list_titles[i_d]
            text = list_texts[i_d]

            d = docs_add[title]
            name[d] = title
            kind[d] = 0
            c = Counter(text)
            for word, count in c.items():
                w = words_add[word]
                name[w] = word
                kind[w] = 1
                if counts:
                    e = g.add_edge(d, w)
                    ecount[e] = count
                else:
                    for n in range(count):
                        g.add_edge(d, w)

        ## filter word-types with less than n_min counts
        if n_min is not None:
            v_n = g.new_vertex_property("int")
            for v in g.vertices():
                v_n[v] = v.out_degree()

            v_filter = g.new_vertex_property("bool")
            for v in g.vertices():
                if v_n[v] < n_min and g.vp["kind"][v] == 1:
                    v_filter[v] = False
                else:
                    v_filter[v] = True
            g.set_vertex_filter(v_filter)
            g.purge_vertices()
            g.clear_filters()

        self.g = g
        self.words = [g.vp["name"][v] for v in g.vertices() if g.vp["kind"][v] == 1]
        self.documents = [g.vp["name"][v] for v in g.vertices() if g.vp["kind"][v] == 0]

    def fit(
        self,
        overlap: bool = False,
        hierarchical: bool = True,
        B_min: Optional[int] = None,
        n_init: int = 1,
        verbose: bool = False,
    ) -> None:
        """Fit the sbm to the word-document network.

        Args:
            overlap: Overlapping or Non-overlapping groups. Overlapping not implemented yet
            hierarchical: Hierarchical SBM or Flat SBM. Flat SBM not implemented yet.
            Bmin: Pass an option to the graph-tool inference specifying the minimum number of blocks.
            n_init: Number of different initial conditions to run in order to avoid local minimum of MDL.
            verbose: If True, perform graph-tool inference in verbose mode.
        """
        g = self.g
        if g is None:
            print("No data to fit the SBM. Load some data first (make_graph)")
        else:
            if overlap and "count" in g.ep:
                raise ValueError(
                    "When using overlapping SBMs, the graph must be constructed with 'counts=False'"
                )
            clabel = g.vp["kind"]

            state_args = {"clabel": clabel, "pclabel": clabel}
            if "count" in g.ep:
                state_args["eweight"] = g.ep.count

            ## the inference
            mdl = np.inf  #
            for i_n_init in range(n_init):
                state_tmp = gt.minimize_nested_blockmodel_dl(
                    g,
                    deg_corr=True,
                    overlap=overlap,
                    state_args=state_args,
                    B_min=B_min,
                    verbose=verbose,
                )
                mdl_tmp = state_tmp.entropy()
                if mdl_tmp < mdl:
                    mdl = 1.0 * mdl_tmp
                    state = state_tmp.copy()

            self.state = state
            ## minimum description length
            self.mdl = state.entropy()
            L = len(state.levels)
            if L == 2:
                self.L = 1
            else:
                self.L = L - 2

    def plot(self, filename: Optional[str] = None, nedges: int = 1000) -> None:
        """Plot the graph and group structure.

        Args:
            filename: Where to save the plot. If None, will not be saved.
            nedges: Subsample to plot (faster, less memory)
        """
        self.state.draw(
            layout="bipartite",
            output=filename,
            subsample_edges=nedges,
            hshortcuts=1,
            hide=0,
        )

    def topics(self, L: int = 0, n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """Get the `n` most common words for each topic in level `L`.

        Args:
            L: Level of the model hierarchy.
            n: Number of top words to return.

        Returns:
            Keys: topic index
            Values: word-probability pairs
        """
        dict_groups = self.get_groups(L=L)

        Bw = dict_groups["Bw"]
        p_w_tw = dict_groups["p_w_tw"]

        words = self.words

        ## loop over all word-groups
        dict_group_words = {}
        for tw in range(Bw):
            p_w_ = p_w_tw[:, tw]
            ind_w_ = np.argsort(p_w_)[::-1][np.isnan(p_w_).sum() :]
            list_words_tw = []
            for i in ind_w_[:n]:
                if p_w_[i] > 0:
                    list_words_tw += [(words[i], p_w_[i])]
                else:
                    break
            dict_group_words[tw] = list_words_tw
        return dict_group_words

    def topicdist(self, doc_index: int, L: int = 0) -> List[Tuple[int, float]]:
        """Get topic distribution for a specific document.

        Args:
            doc_index: Index of query document
            L: Level of the model hierarchy.

        Returns:
            List of (topic index, probability) pairs
        """
        dict_groups = self.get_groups(L=L)

        p_tw_d = dict_groups["p_tw_d"]
        list_topics_tw = []
        for tw, p_tw in enumerate(p_tw_d[:, doc_index]):
            list_topics_tw += [(tw, p_tw)]
        return list_topics_tw

    def clusters(self, L: int = 0, n: int = 10) -> Dict[int, Tuple[Any, float]]:
        """Get n 'most common' documents from each document cluster.

        Most common refers to largest contribution in group membership vector.
        For the non-overlapping case, each document belongs to one and only one group with prob 1.

        Args:
            doc_index: Index of query document
            L: Level of the model hierarchy.

        Returns:
            Keys are cluster indices, values are pairs of document titles and cluster probabilities
        """
        dict_groups = self.get_groups(L=L)
        Bd = dict_groups["Bd"]
        p_td_d = dict_groups["p_td_d"]

        docs = self.documents
        ## loop over all word-groups
        dict_group_docs = {}
        for td in range(Bd):
            p_d_ = p_td_d[td, :]
            ind_d_ = np.argsort(p_d_)[::-1][
                np.isnan(p_d_).sum() :
            ]  # XXX: modified to ignore NaN's
            list_docs_td = []
            for i in ind_d_[:n]:
                if p_d_[i] > 0:
                    list_docs_td += [(docs[i], p_d_[i])]
                else:
                    break
            dict_group_docs[td] = list_docs_td
        return dict_group_docs

    def clusters_query(self, doc_index: int, L: int = 0) -> List[Tuple[int, Any]]:
        """Get all documents in the same group as the query-document.

        Note: Works only for non-overlapping model.
        For overlapping case, we need something else.

        Args:
            doc_index: Index of query document
            L: Level of the model hierarchy.

        Returns:
            List of document index-title pairs
        """
        dict_groups = self.get_groups(L=L)
        p_td_d = dict_groups["p_td_d"]

        documents = self.documents
        ## loop over all word-groups
        td = np.argmax(p_td_d[:, doc_index])

        list_doc_index_sel = np.where(p_td_d[td, :] == 1)[0]

        list_doc_query = []

        for doc_index_sel in list_doc_index_sel:
            if doc_index != doc_index_sel:
                list_doc_query += [(doc_index_sel, documents[doc_index_sel])]

        return list_doc_query

    def get_groups(self, L: int = 0) -> Dict[str, Union[int, np.ndarray]]:
        """Extract statistics on group membership of nodes form the inferred state.

        Computed state is stored in `self.groups[L]`.

        If `L` exists in `self.groups` return pre-computed value.

        Args:
            L: Level of the model hierarchy.

        Returns:
           dict
            - B_d, int, number of doc-groups
            - B_w, int, number of word-groups
            - p_tw_w, array B_w x V; word-group-membership:
                prob that word-node w belongs to word-group tw: P(tw | w)
            - p_td_d, array B_d x D; doc-group membership:
                prob that doc-node d belongs to doc-group td: P(td | d)
            - p_w_tw, array V x B_w; topic distribution:
                prob of word w given topic tw P(w | tw)
            - p_tw_d, array B_w x d; doc-topic mixtures:
                prob of word-group tw in doc d P(tw | d)
        """

        if L in self.groups:  # Avoid recomputation
            return self.groups[L]

        V = self.get_V()
        D = self.get_D()

        g = self.g
        state = self.state
        state_l = state.project_level(L).copy(overlap=True)
        state_l_edges = state_l.get_edge_blocks()  # labeled half-edges

        counts = "count" in self.g.ep.keys()

        ## count labeled half-edges, group-memberships
        B = state_l.B
        n_wb = np.zeros(
            (V, B)
        )  # number of half-edges incident on word-node w and labeled as word-group tw
        n_db = np.zeros(
            (D, B)
        )  # number of half-edges incident on document-node d and labeled as document-group td
        n_dbw = np.zeros(
            (D, B)
        )  # number of half-edges incident on document-node d and labeled as word-group td

        for e in g.edges():
            z1, z2 = state_l_edges[e]
            v1 = e.source()
            v2 = e.target()
            if counts:
                weight = g.ep["count"][e]
            else:
                weight = 1
            n_db[int(v1), z1] += weight
            n_dbw[int(v1), z2] += weight
            n_wb[int(v2) - D, z2] += weight

        ind_d = np.where(np.sum(n_db, axis=0) > 0)[0]
        Bd = len(ind_d)
        n_db = n_db[:, ind_d]

        ind_w = np.where(np.sum(n_wb, axis=0) > 0)[0]
        Bw = len(ind_w)
        n_wb = n_wb[:, ind_w]

        ind_w2 = np.where(np.sum(n_dbw, axis=0) > 0)[0]
        n_dbw = n_dbw[:, ind_w2]

        ## group-membership distributions
        # group membership of each word-node P(t_w | w)
        p_tw_w = (n_wb / np.sum(n_wb, axis=1)[:, np.newaxis]).T

        # group membership of each doc-node P(t_d | d)
        p_td_d = (n_db / np.sum(n_db, axis=1)[:, np.newaxis]).T

        ## topic-distribution for words P(w | t_w)
        p_w_tw = n_wb / np.sum(n_wb, axis=0)[np.newaxis, :]

        ## Mixture of word-groups into documetns P(t_w | d)
        p_tw_d = (n_dbw / np.sum(n_dbw, axis=1)[:, np.newaxis]).T
        # XXX: zeros in `np.sum(n_dbw, axis=1)` induce NaN's

        result = {}
        result["Bd"] = Bd
        result["Bw"] = Bw
        result["p_tw_w"] = p_tw_w
        result["p_td_d"] = p_td_d
        result["p_w_tw"] = p_w_tw
        result["p_tw_d"] = p_tw_d

        self.groups[L] = result  # Cache in `self.groups`

        return result

    def get_V(self) -> int:
        """Number of unique words."""
        return int(np.sum(self.g.vp["kind"].a == 1))

    def get_D(self) -> int:
        """Number of documents."""
        return int(np.sum(self.g.vp["kind"].a == 0))

    def get_N(self) -> int:
        """Number of edges == tokens."""
        return int(self.g.num_edges())  # no. of types

    def _group_to_group_mixture(self, L: int = 0, norm: bool = True) -> np.ndarray:
        g = self.g
        state = self.state
        state_l = state.project_level(L).copy(overlap=True)
        state_l_edges = state_l.get_edge_blocks()  # labeled half-edges

        ## count labeled half-edges, group-memberships
        B = state_l.B
        n_td_tw = np.zeros((B, B))

        counts = "count" in self.g.ep.keys()

        for e in g.edges():
            z1, z2 = state_l_edges[e]
            if counts:
                n_td_tw[z1, z2] += g.ep["count"][e]
            else:
                n_td_tw[z1, z2] += 1

        ind_d = np.where(np.sum(n_td_tw, axis=1) > 0)[0]
        Bd = len(ind_d)

        n_td_tw = n_td_tw[:Bd, Bd:]
        if norm is True:
            return n_td_tw / np.sum(n_td_tw)
        else:
            return n_td_tw

    def pmi_td_tw(self, L: int = 0) -> np.ndarray:
        """Point-wise mutual information between topic-groups and doc-groups, S(td,tw).


        It corresponds to $S(td,tw) = log P(tw | td) / \tilde{P}(tw | td)$.

        This is the log-ratio between
        P(tw | td) == prb of topic tw in doc-group td;
        $\tilde{P}(tw | td) = P(tw)$ expected prob of topic tw in doc-group td under random null model.

        Args:
            L: Level of the model hierarchy.

        Returns:
            Array of shape Bd x Bw.
        """
        p_td_tw = self._group_to_group_mixture(L=L)
        p_tw_td = p_td_tw.T
        p_td = np.sum(p_tw_td, axis=0)
        p_tw = np.sum(p_tw_td, axis=1)
        pmi_td_tw = np.log(p_tw_td / (p_td * p_tw[:, np.newaxis])).T
        return pmi_td_tw
