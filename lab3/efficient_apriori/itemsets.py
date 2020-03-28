import itertools
import collections
import collections.abc
import numbers
import typing
from abc import ABC, abstractmethod

from collections import defaultdict
from dataclasses import field, dataclass


@dataclass
class ItemsetCount:
    itemset_count: int = 0
    members: set = field(default_factory=set)

    def increment_count(self, transaction_id: int):
        self.itemset_count += 1
        self.members.add(transaction_id)


class _ItemsetCounter(ABC):
    @abstractmethod
    def itemset_counter(self):
        pass

    @abstractmethod
    def get_count(self, count):
        pass

    @abstractmethod
    def singleton_itemsets(self, get_transactions):
        pass

    @abstractmethod
    def large_itemsets(self, counts, min_support, num_transactions):
        pass

    @abstractmethod
    def candidate_itemset_counts(
        self, C_k, C_k_sets, counter, counts, row, transaction
    ):
        pass


class _Counter(_ItemsetCounter):
    def itemset_counter(self):
        return 0

    def get_count(self, count):
        return count

    def singleton_itemsets(self, get_transactions):
        counts = defaultdict(self.itemset_counter)
        num_transactions = 0
        for _, transaction in get_transactions():
            num_transactions += 1
            for item in transaction:
                counts[item] += 1
        return counts, num_transactions

    def large_itemsets(self, counts, min_support, num_transactions):
        return [
            (i, c)
            for (i, c) in counts.items()
            if (c / num_transactions) >= min_support
        ]

    def candidate_itemset_counts(
        self, C_k, C_k_sets, counter, counts, row, transaction
    ):
        # Assert that no items were found in this row
        found_any = False
        issubset = set.issubset  # Micro-optimization
        for candidate, candidate_set in zip(C_k, C_k_sets):
            # This is where most of the time is spent in the algorithm
            # If the candidate set is a subset, add count and mark the row
            if issubset(candidate_set, transaction):
                counts[candidate] += 1
                found_any = True
        return counts, found_any


class _CounterWithIds(_ItemsetCounter):
    def itemset_counter(self):
        return ItemsetCount()

    def get_count(self, count):
        return count.itemset_count

    def singleton_itemsets(self, get_transactions):
        counts = defaultdict(self.itemset_counter)
        num_transactions = 0
        for row, transaction in get_transactions():
            num_transactions += 1
            for item in transaction:
                counts[item].increment_count(row)
        return counts, num_transactions

    def large_itemsets(self, counts, min_support, num_transactions):
        return [
            (i, count)
            for (i, count) in counts.items()
            if (count.itemset_count / num_transactions) >= min_support
        ]

    def candidate_itemset_counts(
        self, C_k, C_k_sets, counter, counts, row, transaction
    ):
        # Assert that no items were found in this row
        found_any = False
        issubset = set.issubset  # Micro-optimization
        for candidate, candidate_set in zip(C_k, C_k_sets):
            # This is where most of the time is spent in the algorithm
            # If the candidate set is a subset, add count and mark the row
            if issubset(candidate_set, transaction):
                counts[candidate].increment_count(row)
                found_any = True
        return counts, found_any


def join_step(itemsets: typing.List[tuple]):
    """
    Join k length itemsets into k + 1 length itemsets.
    This algorithm assumes that the list of itemsets are sorted, and that the
    itemsets themselves are sorted tuples. Instead of always enumerating all
    n^2 combinations, the algorithm only has n^2 runtime for each block of
    itemsets with the first k - 1 items equal.
    Parameters
    ----------
    itemsets : list of itemsets
        A list of itemsets of length k, to be joined to k + 1 length
        itemsets.
    Examples
    --------
    >>> # This is an example from the 1994 paper by Agrawal et al.
    >>> itemsets = [(1, 2, 3), (1, 2, 4), (1, 3, 4), (1, 3, 5), (2, 3, 4)]
    >>> list(join_step(itemsets))
    [(1, 2, 3, 4), (1, 3, 4, 5)]
    """
    i = 0
    # Iterate over every itemset in the itemsets
    while i < len(itemsets):

        # The number of rows to skip in the while-loop, initially set to 1
        skip = 1

        # Get all but the last item in the itemset, and the last item
        *itemset_first, itemset_last = itemsets[i]

        # We now iterate over every itemset following this one, stopping
        # if the first k - 1 items are not equal. If we're at (1, 2, 3),
        # we'll consider (1, 2, 4) and (1, 2, 7), but not (1, 3, 1)

        # Keep a list of all last elements, i.e. tail elements, to perform
        # 2-combinations on later on
        tail_items = [itemset_last]
        tail_items_append = tail_items.append  # Micro-optimization

        # Iterate over ever itemset following this itemset
        for j in range(i + 1, len(itemsets)):

            # Get all but the last item in the itemset, and the last item
            *itemset_n_first, itemset_n_last = itemsets[j]

            # If it's the same, append and skip this itemset in while-loop
            if itemset_first == itemset_n_first:

                # Micro-optimization
                tail_items_append(itemset_n_last)
                skip += 1

            # If it's not the same, break out of the for-loop
            else:
                break

        # For every 2-combination in the tail items, yield a new candidate
        # itemset, which is sorted.
        itemset_first_tuple = tuple(itemset_first)
        for a, b in sorted(itertools.combinations(tail_items, 2)):
            yield itemset_first_tuple + (a,) + (b,)

        # Increment the while-loop counter
        i += skip


def prune_step(
    itemsets: typing.Iterable[tuple], possible_itemsets: typing.List[tuple]
):
    """
    Prune possible itemsets whose subsets are not in the list of itemsets.
    Parameters
    ----------
    itemsets : list of itemsets
        A list of itemsets of length k.
    possible_itemsets : list of itemsets
        A list of possible itemsets of length k + 1 to be pruned.
    Examples
    -------
    >>> itemsets = [('a', 'b', 'c'), ('a', 'b', 'd'),
    ...             ('b', 'c', 'd'), ('a', 'c', 'd')]
    >>> possible_itemsets = list(join_step(itemsets))
    >>> list(prune_step(itemsets, possible_itemsets))
    [('a', 'b', 'c', 'd')]
    """

    # For faster lookups
    itemsets = set(itemsets)

    # Go through every possible itemset
    for possible_itemset in possible_itemsets:

        # Remove 1 from the combination, same as k-1 combinations
        # The itemsets created by removing the last two items in the possible
        # itemsets must be part of the itemsets by definition,
        # due to the way the `join_step` function merges the sorted itemsets

        for i in range(len(possible_itemset) - 2):
            removed = possible_itemset[:i] + possible_itemset[i + 1 :]

            # If every k combination exists in the set of itemsets,
            # yield the possible itemset. If it does not exist, then it's
            # support cannot be large enough, since supp(A) >= supp(AB) for
            # all B, and if supp(S) is large enough, then supp(s) must be large
            # enough for every s which is a subset of S.
            # This is the downward-closure property of the support function.
            if removed not in itemsets:
                break

        # If we have not breaked yet
        else:
            yield possible_itemset


def apriori_gen(itemsets: typing.List[tuple]):
    """
    Compute all possible k + 1 length supersets from k length itemsets.
    This is done efficiently by using the downward-closure property of the
    support function, which states that if support(S) > k, then support(s) > k
    for every subset s of S.
    Parameters
    ----------
    itemsets : list of itemsets
        A list of itemsets of length k.
    Examples
    -------
    >>> # This is an example from the 1994 paper by Agrawal et al.
    >>> itemsets = [(1, 2, 3), (1, 2, 4), (1, 3, 4), (1, 3, 5), (2, 3, 4)]
    >>> possible_itemsets = list(join_step(itemsets))
    >>> list(prune_step(itemsets, possible_itemsets))
    [(1, 2, 3, 4)]
    """
    possible_extensions = join_step(itemsets)
    yield from prune_step(itemsets, possible_extensions)


def itemsets_from_transactions(
    transactions: typing.Union[typing.List[tuple], typing.Callable],
    min_support: float,
    max_length: int = 8,
    verbosity: int = 0,
    output_transaction_ids: bool = False,
):
    """
    Compute itemsets from transactions by building the itemsets bottom up and
    iterating over the transactions to compute the support repedately. This is
    the heart of the Apriori algorithm by Agrawal et al. in the 1994 paper.
    Parameters
    ----------
    transactions : a list of itemsets (tuples with hashable entries),
                   or a function returning a generator
        A list of transactions. They can be of varying size. To pass through
        data without reading everything into memory at once, a callable
        returning a generator may also be passed.
    min_support : float
        The minimum support of the itemsets, i.e. the minimum frequency as a
        percentage.
    max_length : int
        The maximum length of the itemsets.
    verbosity : int
        The level of detail printing when the algorithm runs. Either 0, 1 or 2.
    output_transaction_ids : bool
        If set to true, the output contains the ids of transactions that
        contain a frequent itemset. The ids are the enumeration of the
        transactions in the sequence they appear.
    Examples
    --------
    >>> # This is an example from the 1994 paper by Agrawal et al.
    >>> transactions = [(1, 3, 4), (2, 3, 5), (1, 2, 3, 5), (2, 5)]
    >>> itemsets, _ = itemsets_from_transactions(transactions, min_support=2/5)
    >>> itemsets[1] == {(1,): 2, (2,): 3, (3,): 3, (5,): 3}
    True
    >>> itemsets[2] == {(1, 3): 2, (2, 3): 2, (2, 5): 3, (3, 5): 2}
    True
    >>> itemsets[3] == {(2, 3, 5): 2}
    True
    """

    # STEP 0 - Sanitize user inputs
    # -----------------------------
    if not (
        isinstance(min_support, numbers.Number) and (0 <= min_support <= 1)
    ):
        raise ValueError("`min_support` must be a number between 0 and 1.")

    counter: typing.Union[_CounterWithIds, _Counter]  # Type info for mypy
    counter = (
        _CounterWithIds()
        if (transactions and output_transaction_ids)
        else _Counter()
    )

    wrong_transaction_type_msg = (
        "`transactions` must be an iterable or a "
        "callable returning an iterable."
    )

    if not transactions:
        return dict(), 0  # large_itemsets, num_transactions

    if isinstance(transactions, collections.abc.Iterable):

        def transaction_rows():
            for count, t in enumerate(transactions):
                yield count, set(t)

    # Assume the transactions is a callable, returning a generator
    elif callable(transactions):

        def transaction_rows():
            for count, t in enumerate(transactions()):
                yield count, set(t)

        if not isinstance(transactions(), collections.abc.Generator):
            raise TypeError(wrong_transaction_type_msg)
    else:
        raise TypeError(wrong_transaction_type_msg)

    # Keep a dictionary stating whether to consider the row, this will allow
    # row-pruning later on if no information was retrieved earlier from it
    use_transaction: typing.DefaultDict[int, bool] = defaultdict(lambda: True)

    # STEP 1 - Generate all large itemsets of size 1
    # ----------------------------------------------
    if verbosity > 0:
        print("Generating itemsets.")
        print(" Counting itemsets of length 1.")

    counts, num_transactions = counter.singleton_itemsets(transaction_rows)

    large_itemsets = counter.large_itemsets(
        counts, min_support, num_transactions
    )

    if verbosity > 0:
        num_cand, num_itemsets = len(counts.items()), len(large_itemsets)
        print("  Found {} candidate itemsets of length 1.".format(num_cand))
        print("  Found {} large itemsets of length 1.".format(num_itemsets))
    if verbosity > 1:
        print("    {}".format(list((i,) for (i, counts) in large_itemsets)))

    # If large itemsets were found, convert to dictionary
    if large_itemsets:
        large_itemsets = {
            1: {(i,): counts for (i, counts) in (large_itemsets)}
        }
    # No large itemsets were found, return immediately
    else:
        return dict(), 0  # large_itemsets, num_transactions

    # STEP 2 - Build up the size of the itemsets
    # ------------------------------------------

    # While there are itemsets of the previous size
    k = 2
    while large_itemsets[k - 1] and (max_length != 1):
        if verbosity > 0:
            print(" Counting itemsets of length {}.".format(k))

        # STEP 2a) - Build up candidate of larger itemsets

        # Retrieve the itemsets of the previous size, i.e. of size k - 1
        # They must be sorted to maintain the invariant when joining/pruning
        itemsets_list = sorted(large_itemsets[k - 1].keys())

        # Gen candidates of length k + 1 by joining, prune, and copy as set
        C_k = list(apriori_gen(itemsets_list))
        C_k_sets = [set(itemset) for itemset in C_k]

        if verbosity > 0:
            print(
                "  Found {} candidate itemsets of length {}.".format(
                    len(C_k), k
                )
            )
        if verbosity > 1:
            print("   {}".format(C_k))

        # If no candidate itemsets were found, break out of the loop
        if not C_k:
            break

        # Prepare counts of candidate itemsets (from the prune step)
        counts = defaultdict(counter.itemset_counter)
        if verbosity > 1:
            print("    Iterating over transactions.")
        for row, transaction in transaction_rows():

            # If we've excluded this transaction earlier, do not consider it
            if not use_transaction[row]:
                continue

            counts, found_any = counter.candidate_itemset_counts(
                C_k, C_k_sets, counter, counts, row, transaction
            )

            # If no candidate sets were found in this row, skip this row of
            # transactions in the future
            if not found_any:
                use_transaction[row] = False

        # Only keep the candidates whose support is over the threshold
        C_k = counter.large_itemsets(counts, min_support, num_transactions)

        # If no itemsets were found, break out of the loop
        if not C_k:
            break

        # Candidate itemsets were found, add them and progress the while-loop
        large_itemsets[k] = {i: counts for (i, counts) in C_k}

        if verbosity > 0:
            num_found = len(large_itemsets[k])
            pp = "  Found {} large itemsets of length {}.".format(num_found, k)
            print(pp)
        if verbosity > 1:
            print("   {}".format(list(large_itemsets[k].keys())))
        k += 1

        # Break out if we are about to consider larger itemsets than the max
        if k > max_length:
            break

    if verbosity > 0:
        print("Itemset generation terminated.\n")

    return large_itemsets, num_transactions


class Rule(object):
    """
    A class for a rule.
    """

    # Number of decimals used for printing
    _decimals = 3

    def __init__(
        self,
        lhs: tuple,
        rhs: tuple,
        count_full: int = 0,
        count_lhs: int = 0,
        count_rhs: int = 0,
        num_transactions: int = 0,
    ):
        """
        Initialize a new rule. This call is a thin wrapper around some data.
        Parameters
        ----------
        lhs : tuple
            The left hand side (antecedent) of the rule. Each item in the tuple
            must be hashable, e.g. a string or an integer.
        rhs : tuple
            The right hand side (consequent) of the rule.
        count_full : int
            The count of the union of the lhs and rhs in the dataset.
        count_lhs : int
            The count of the lhs in the dataset.
        count_rhs : int
            The count of the rhs in the dataset.
        num_transactions : int
            The number of transactions in the dataset.
        Examples
        --------
        >>> r = Rule(('a', 'b'), ('c',), 50, 100, 150, 200)
        >>> r.confidence  # Probability of 'c', given 'a' and 'b'
        0.5
        >>> r.support  # Probability of ('a', 'b', 'c') in the data
        0.25
        >>> # Ratio of observed over expected support if lhs, rhs = independent
        >>> r.lift == 2 / 3
        True
        >>> print(r)
        {a, b} -> {c} (conf: 0.500, supp: 0.250, lift: 0.667, conv: 0.500)
        >>> r
        {a, b} -> {c}
        """
        self.lhs = lhs  # antecedent
        self.rhs = rhs  # consequent
        self.count_full = count_full
        self.count_lhs = count_lhs
        self.count_rhs = count_rhs
        self.num_transactions = num_transactions

    @property
    def confidence(self):
        """
        The confidence of a rule is the probability of the rhs given the lhs.
        If X -> Y, then the confidence is P(Y|X).
        """
        try:
            return self.count_full / self.count_lhs
        except ZeroDivisionError:
            return None
        except AttributeError:
            return None

    @property
    def support(self):
        """
        The support of a rule is the frequency of which the lhs and rhs appear
        together in the dataset. If X -> Y, then the support is P(Y and X).
        """
        try:
            return self.count_full / self.num_transactions
        except ZeroDivisionError:
            return None
        except AttributeError:
            return None

    @property
    def lift(self):
        """
        The lift of a rule is the ratio of the observed support to the expected
        support if the lhs and rhs were independent.If X -> Y, then the lift is
        given by the fraction P(X and Y) / (P(X) * P(Y)).
        """
        try:
            observed_support = self.count_full / self.num_transactions
            prod_counts = self.count_lhs * self.count_rhs
            expected_support = prod_counts / self.num_transactions ** 2
            return observed_support / expected_support
        except ZeroDivisionError:
            return None
        except AttributeError:
            return None

    @property
    def conviction(self):
        """
        The conviction of a rule X -> Y is the ratio P(not Y) / P(not Y | X).
        It's the proportion of how often Y does not appear in the data to how
        often Y does not appear in the data, given X. If the ratio is large,
        then the confidence is large and Y appears often.
        """
        try:
            eps = 10e-10  # Avoid zero division
            prob_not_rhs = 1 - self.count_rhs / self.num_transactions
            prob_not_rhs_given_lhs = 1 - self.confidence
            return prob_not_rhs / (prob_not_rhs_given_lhs + eps)
        except ZeroDivisionError:
            return None
        except AttributeError:
            return None

    @property
    def rpf(self):
        """
        The RPF (Rule Power Factor) is the confidence times the support.
        """
        try:
            return self.confidence * self.support
        except ZeroDivisionError:
            return None
        except AttributeError:
            return None

    @staticmethod
    def _pf(s):
        """
        Pretty formatting of an iterable.
        """
        return "{" + ", ".join(str(k) for k in s) + "}"

    def __repr__(self):
        """
        Representation of a rule.
        """
        return "{} -> {}".format(self._pf(self.lhs), self._pf(self.rhs))

    def __str__(self):
        """
        Printing of a rule.
        """
        conf = "conf: {0:.3f}".format(self.confidence)
        supp = "supp: {0:.3f}".format(self.support)
        lift = "lift: {0:.3f}".format(self.lift)
        conv = "conv: {0:.3f}".format(self.conviction)

        return "{} -> {} ({}, {}, {}, {})".format(
            self._pf(self.lhs), self._pf(self.rhs), conf, supp, lift, conv
        )

    def __eq__(self, other):
        """
        Equality of two rules.
        """
        return (set(self.lhs) == set(other.lhs)) and (
            set(self.rhs) == set(other.rhs)
        )

    def __hash__(self):
        """
        Hashing a rule for efficient set and dict representation.
        """
        return hash(frozenset(self.lhs + self.rhs))

    def __len__(self):
        """
        The length of a rule, defined as the number of items in the rule.
        """
        return len(self.lhs + self.rhs)


def generate_rules_simple(
    itemsets: typing.Dict[int, typing.Dict],
    min_confidence: float,
    num_transactions: int,
):
    """
    DO NOT USE. This is a simple top-down algorithm for generating association
    rules. It is included here for testing purposes, and because it is
    mentioned in the 1994 paper by Agrawal et al. It is slow because it does
    not enumerate the search space efficiently: it produces duplicates, and it
    does not prune the search space efficiently.
    Simple algorithm for generating association rules from itemsets.
    """

    # Iterate over every size
    for size in itemsets.keys():

        # Do not consider itemsets of size 1
        if size < 2:
            continue

        # This algorithm returns duplicates, so we keep track of items yielded
        # in a set to avoid yielding duplicates
        yielded: set = set()
        yielded_add = yielded.add

        # Iterate over every itemset of the prescribed size
        for itemset in itemsets[size].keys():

            # Generate rules
            for result in _genrules(
                itemset, itemset, itemsets, min_confidence, num_transactions
            ):

                # If the rule has been yieded, keep going, else add and yield
                if result in yielded:
                    continue
                else:
                    yielded_add(result)
                    yield result


def _genrules(l_k, a_m, itemsets, min_conf, num_transactions):
    """
    DO NOT USE. This is the gen-rules algorithm from the 1994 paper by Agrawal
    et al. It's a subroutine called by `generate_rules_simple`. However, the
    algorithm `generate_rules_simple` should not be used.
    The naive algorithm from the original paper.
    Parameters
    ----------
    l_k : tuple
        The itemset containing all elements to be considered for a rule.
    a_m : tuple
        The itemset to take m-length combinations of, an move to the left of
        l_k. The itemset a_m is a subset of l_k.
    """

    def count(itemset):
        """
        Helper function to retrieve the count of the itemset in the dataset.
        """
        return itemsets[len(itemset)][itemset]

    # Iterate over every k - 1 combination of a_m to produce
    # rules of the form a -> (l - a)
    for a_m in itertools.combinations(a_m, len(a_m) - 1):

        # Compute the count of this rule, which is a_m -> (l_k - a_m)
        confidence = count(l_k) / count(a_m)

        # Keep going if the confidence level is too low
        if confidence < min_conf:
            continue

        # Create the right hand set: rhs = (l_k - a_m) , and keep it sorted
        rhs = set(l_k).difference(set(a_m))
        rhs = tuple(sorted(rhs))

        # Create new rule object and yield it
        yield Rule(
            a_m, rhs, count(l_k), count(a_m), count(rhs), num_transactions
        )

        # If the left hand side has one item only, do not recurse the function
        if len(a_m) <= 1:
            continue
        yield from _genrules(l_k, a_m, itemsets, min_conf, num_transactions)


def generate_rules_apriori(
    itemsets: typing.Dict[int, typing.Dict[tuple, int]],
    min_confidence: float,
    num_transactions: int,
    verbosity: int = 0,
):
    """
    Bottom up algorithm for generating association rules from itemsets, very
    similar to the fast algorithm proposed in the original 1994 paper by
    Agrawal et al.
    The algorithm is based on the observation that for {a, b} -> {c, d} to
    hold, both {a, b, c} -> {d} and {a, b, d} -> {c} must hold, since in
    general conf( {a, b, c} -> {d} ) >= conf( {a, b} -> {c, d} ).
    In other words, if either of the two one-consequent rules do not hold, then
    there is no need to ever consider the two-consequent rule.
    Parameters
    ----------
    itemsets : dict of dicts
        The first level of the dictionary is of the form (length, dict of item
        sets). The second level is of the form (itemset, count_in_dataset)).
    min_confidence :  float
        The minimum confidence required for the rule to be yielded.
    num_transactions : int
        The number of transactions in the data set.
    verbosity : int
        The level of detail printing when the algorithm runs. Either 0, 1 or 2.
    Examples
    --------
    >>> itemsets = {1: {('a',): 3, ('b',): 2, ('c',): 1},
    ...             2: {('a', 'b'): 2, ('a', 'c'): 1}}
    >>> list(generate_rules_apriori(itemsets, 1.0, 3))
    [{b} -> {a}, {c} -> {a}]
    """
    # Validate user inputs
    if not (
        (0 <= min_confidence <= 1)
        and isinstance(min_confidence, numbers.Number)
    ):
        raise ValueError("`min_confidence` must be a number between 0 and 1.")

    if not (
        (num_transactions >= 0)
        and isinstance(num_transactions, numbers.Number)
    ):
        raise ValueError("`num_transactions` must be a number greater than 0.")

    def count(itemset):
        """
        Helper function to retrieve the count of the itemset in the dataset.
        """
        return itemsets[len(itemset)][itemset]

    if verbosity > 0:
        print("Generating rules from itemsets.")

    # For every itemset of a perscribed size
    for size in itemsets.keys():

        # Do not consider itemsets of size 1
        if size < 2:
            continue

        if verbosity > 0:
            print(" Generating rules of size {}.".format(size))

        # For every itemset of this size
        for itemset in itemsets[size].keys():

            # Special case to capture rules such as {others} -> {1 item}
            for removed in itertools.combinations(itemset, 1):

                # Compute the left hand side
                remaining = set(itemset).difference(set(removed))
                lhs = tuple(sorted(remaining))

                # If the confidence is high enough, yield the rule
                conf = count(itemset) / count(lhs)
                if conf >= min_confidence:
                    yield Rule(
                        lhs,
                        removed,
                        count(itemset),
                        count(lhs),
                        count(removed),
                        num_transactions,
                    )

            # Generate combinations to start off of. These 1-combinations will
            # be merged to 2-combinations in the function `_ap_genrules`
            H_1 = list(itertools.combinations(itemset, 1))
            yield from _ap_genrules(
                itemset, H_1, itemsets, min_confidence, num_transactions
            )

    if verbosity > 0:
        print("Rule generation terminated.\n")


def _ap_genrules(
    itemset: tuple,
    H_m: typing.List[tuple],
    itemsets: typing.Dict[int, typing.Dict[tuple, int]],
    min_conf: float,
    num_transactions: int,
):
    """
    Recursively build up rules by adding more items to the right hand side.
    This algorithm is called `ap-genrules` in the original paper. It is
    called by the `generate_rules_apriori` generator above. See it's docs.
    Parameters
    ----------
    itemset : tuple
        The itemset under consideration.
    H_m : tuple
        Subsets of the itemset of length m, to be considered for rhs of a rule.
    itemsets : dict of dicts
        All itemsets and counts for in the data set.
    min_conf : float
        The minimum confidence for a rule to be returned.
    num_transactions : int
        The number of transactions in the data set.
    """

    def count(itemset):
        """
        Helper function to retrieve the count of the itemset in the dataset.
        """
        return itemsets[len(itemset)][itemset]

    # If H_1 is so large that calling `apriori_gen` will produce right-hand
    # sides as large as `itemset`, there will be no right hand side
    # This cannot happen, so abort if it will
    if len(itemset) <= (len(H_m[0]) + 1):
        return

    # Generate left-hand itemsets of length k + 1 if H is of length k
    H_m = list(apriori_gen(H_m))
    H_m_copy = H_m.copy()

    # For every possible right hand side
    for h_m in H_m:
        # Compute the right hand side of the rule
        lhs = tuple(sorted(set(itemset).difference(set(h_m))))

        # If the confidence is high enough, yield the rule, else remove from
        # the upcoming recursive generator call
        if (count(itemset) / count(lhs)) >= min_conf:
            yield Rule(
                lhs,
                h_m,
                count(itemset),
                count(lhs),
                count(h_m),
                num_transactions,
            )
        else:
            H_m_copy.remove(h_m)

    # Unless the list of right-hand sides is empty, recurse the generator call
    if H_m_copy:
        yield from _ap_genrules(
            itemset, H_m_copy, itemsets, min_conf, num_transactions
        )