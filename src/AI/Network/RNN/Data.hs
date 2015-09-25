{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts #-}
-----------------------------------------------------------------------------
--
-- Module      :  AI.Network.RNN.Data
-- Copyright   :  (c) JP Moresmau
-- License     :  BSD3
--
-- Maintainer  :  JP Moresmau <jp@moresmau.fr>
-- Stability   :  experimental
-- Portability :
--
-- | Conversion from and to text data
--
-----------------------------------------------------------------------------

module AI.Network.RNN.Data  where

import qualified Data.Text as T
import qualified Data.Map as DM
import qualified Data.Set as DS
import Data.Maybe
import Data.Tuple
import Data.List
import Control.Monad.Random as R
import Control.Monad
import Numeric.LinearAlgebra.HMatrix  as M
import Data.Foldable as F (toList)

import qualified Data.SDR as SDR
import qualified Data.IntSet as I

import AI.Network.RNN.Types
import AI.Network.RNN.Util
-- import Debug.Trace


-- | Transform text into training data
-- the encoding of each Char is equilateral encoding
textToTrainData :: T.Text -> TrainData ((DM.Map Int Char),Matrix Double)
textToTrainData t =
    let (ids,m) = T.foldl' toIDs ([],DM.empty) t
        sz = DM.size m
        em = equilateralEncoding sz
        is = map (toArr em) $ reverse ids
        charMap = DM.fromList $ map swap $ DM.assocs m
    in
        TrainData (M.fromList (replicate (sz-1) 0): init is) is (sz-1) (charMap,em)
    where
        toIDs :: ([Int],DM.Map Char Int) -> Char -> ([Int],DM.Map Char Int)
        toIDs (ids,m) c =
            let mmyid = DM.lookup c m
            in case mmyid of
                Just myid -> (myid:ids,m)
                Nothing   ->
                    let myid = DM.size m
                    in (myid:ids,DM.insert c myid m)
--        toArr :: Int -> Int -> Vector Double
--        toArr sz idx = M.fromList $ replicate idx 0 ++ [1] ++ replicate (sz-idx-1) 0
        toArr em idx = flatten $ em ? [idx]

-- | Decode the data from the equilateral encoding
dataToText :: ((DM.Map Int Char),Matrix Double) -> [Vector Double] -> T.Text
dataToText (m,em)  = T.pack . F.toList . fmap toC
    where
        toC :: Vector Double -> Char
        -- toC ds = fromJust $ DM.lookup (maxIndex ds) m
        toC ds = fromJust $ DM.lookup (equilateralDecoding em ds) m

-- | Generate random text with the probabilities given by the vector (not used)
randDataToText :: RandomGen g => DM.Map Int Char -> [Vector Double] -> Rand g T.Text
randDataToText m  = liftM T.pack . mapM (toC . M.toList)
    where
        toC :: RandomGen g => [Double] -> Rand g Char
        toC ls = do
            let m1 = normalize $ map (\(ix,d)->(fromJust $ DM.lookup ix m,d)) $ zip [0..] ls
                m2 = map (\(a,b)->(a,toRational b)) m1
                -- s  = sum $ map (exp . snd) m1
                -- m3 = map (\(a,b)-> (a,exp(b)/s)) m1
            R.fromList m2
            --return $ fst $ last $ sortBy (comparing snd) m3

-- | Generate text using equilateral encoding and a given network
generate :: (RandomGen g,RNNEval a)
    => ((DM.Map Int Char),Matrix Double) -- ^ Character map
    -> Int  -- ^ Number of characters to generate
    -> Int -- ^ Size of the vectors for each characters
    -> a -- network
    -> Rand g T.Text
generate m nb sz rnn =generate' m nb sz rnn dataToText

-- | Generate text using sparse encoding and a given network
generateS :: (RandomGen g,RNNEval a)
    => SDR.SDRSet Char -- ^ Character map
    -> Int  -- ^ Number of characters to generate
    -> Int -- ^ Size of the vectors for each characters
    -> a -- network
    -> Rand g T.Text
generateS m nb sz rnn =generate' m nb sz rnn dataToTextS

-- | Generate text given a decoding function
-- current implementation is not random!
generate' :: (RandomGen g,RNNEval a) => m -> Int -> Int -> a ->
    (m -> [Vector Double] -> T.Text) -> Rand g T.Text
generate' m nb sz rnn f =  do
    let (_,_,alls) = foldl' go (rnn,M.fromList $ replicate sz 0,[]) [1..nb]
    --randDataToText m alls
    return $ f m alls
    where
        go :: (RNNEval a) => (a,Vector Double,[Vector Double]) -> Int -> (a,Vector Double,[Vector Double])
        go (rnn1,is1,oss) _ =
            let (rnn2,os) = evalStep rnn1 is1
                os2 = norm os
            in (rnn2,os2,oss ++ [os])
        norm :: Vector Double -> Vector Double
        norm os = let
            mx = maximum $ M.toList os
            --in M.assoc (M.size os) 0 [(mx,1)]
            in M.fromList $ map (\a->if a == mx then 1 else 0) $ M.toList os

-- | Train data using sparse encoding: 2 values in the vector are set to one for each character
-- This was a test to try to reduce the size of the network but is not efficient
textToTrainDataS :: T.Text -> TrainData (SDR.SDRSet Char)
textToTrainDataS t =
    let charSet = T.foldl' (flip DS.insert) DS.empty t
        sz  = DS.size charSet
        markers = 1
        total = sz * markers
        sdr = SDR.build (sz*markers) markers charSet
        is = T.foldr (\c l->enc total sdr c : l) [] t
    in TrainData (M.fromList (replicate total 0): init is) is total sdr
    where
        enc :: Int -> SDR.SDRSet Char -> Char -> Vector Double
        enc total sdr c =
            let is=SDR.encode sdr c
            in M.assoc total 0 $ zip (I.toList is) (repeat 1)

--textToTrainDataS t =
--    let (ids,m) = T.foldl' toIDs ([],DM.empty) t
--        sz = sparseSize $ DM.size m
--        spr = DM.fromList $ zip [0..] $ sparse $ DM.size m
--        is = map (toArr spr sz) $ reverse ids
--        charMap = DM.fromList $ map (\(c,i)->(fromJust $ DM.lookup i spr,c)) $ DM.assocs m
--    in
--        TrainData (M.fromList (replicate sz 0): init is) is sz charMap
--    where
--        toIDs :: ([Int],DM.Map Char Int) -> Char -> ([Int],DM.Map Char Int)
--        toIDs (ids,m) c =
--            let mmyid = DM.lookup c m
--            in case mmyid of
--                Just myid -> (myid:ids,m)
--                Nothing   ->
--                    let myid = DM.size m
--                    in (myid:ids,DM.insert c myid m)
--        toArr :: DM.Map Int (Int,Int) -> Int -> Int -> Vector Double
--        toArr m sz idx = let
--            (i1,i2) = fromJust $ DM.lookup idx m
--            in M.fromList $ replicate i1 0 ++ [1] ++ replicate (i2-i1-1) 0 ++ [1] ++ replicate (sz-i2-1) 0

-- | Generate text from sparse encoding data
dataToTextS :: SDR.SDRSet Char -> [Vector Double] -> T.Text
dataToTextS sdr = T.pack . map b
    where b v =
            let is = I.fromList $ map fst $ filter (\(_,c)->c>=0.5) $ zipWith (\a b->(b,a)) (M.toList v) [0..]
            in SDR.best sdr is
--dataToTextS m  = T.pack . F.toList . fmap toC
--    where
--        toC :: Vector Double -> Char
--        toC ds = let
--            ix1 = maxIndex ds
--            ix2 = maxIndex $ accum ds const [(ix1,0)]
--            tpl = if ix1 < ix2
--                    then (ix1,ix2)
--                    else (ix2,ix1)
--            mc = DM.lookup tpl m
--            in fromMaybe (fromJust $ DM.lookup (0,1) m) mc
--         --fromJust $ DM.lookup (maxIndex ds) m
