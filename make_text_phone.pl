#!/usr/bin/env perl
# make_text_phone.pl
# Usage: make_text_phone.pl text lexicon.txt > text-phone

my ($text, $lexicon) = @ARGV;
open(LX, "<", $lexicon) or die "Could not open $lexicon\n";
my %lex = ();
while (<LX>) {
    chomp;
    my ($word, @phones) = split;
    $lex{$word} = join(" ", @phones);
}
close(LX);

open(TX, "<", $text) or die "Could not open $text\n";
while (<TX>) {
    chomp;
    my ($uttid, @words) = split;
    for (my $i = 0; $i < @words; $i++) {
        my $wid = $uttid . "." . $i;
        if (exists $lex{$words[$i]}) {
            print "$wid $lex{$words[$i]}\n";
        } else {
            warn "Word $words[$i] not in lexicon!\n";
        }
    }
}
close(TX);

