#!/usr/bin/env perl
# Script to remove phone markers for GOP computation
# This creates a mapping from phones with markers to pure phones

use strict;
use warnings;

if (@ARGV != 3) {
    print STDERR "Usage: $0 <phones-with-markers> <pure-phones> <output-mapping>\n";
    print STDERR "  phones-with-markers: phones.txt from graph\n";
    print STDERR "  pure-phones: phones-pure.txt file\n";
    print STDERR "  output-mapping: phone-to-pure-phone.int mapping\n";
    exit(1);
}

my ($phones_file, $pure_phones_file, $output_file) = @ARGV;

# Read pure phones
my %pure_phones = ();
open(my $pure_fh, '<', $pure_phones_file) or die "Cannot open $pure_phones_file: $!";
while (my $line = <$pure_fh>) {
    chomp $line;
    my ($phone, $id) = split(/\s+/, $line);
    $pure_phones{$phone} = $id;
}
close($pure_fh);

# Read phones with markers and create mapping
open(my $phones_fh, '<', $phones_file) or die "Cannot open $phones_file: $!";
open(my $output_fh, '>', $output_file) or die "Cannot open $output_file: $!";

while (my $line = <$phones_fh>) {
    chomp $line;
    my ($phone, $id) = split(/\s+/, $line);

    # Skip phone ID 0 (epsilon/silence marker)
    next if $id == 0;

    # Remove markers like _B, _E, _I, _S from phone
    my $pure_phone = $phone;
    $pure_phone =~ s/_[BEIS]$//;

    # Map to pure phone ID if it exists
    if (exists $pure_phones{$pure_phone}) {
        print $output_fh "$id $pure_phones{$pure_phone}\n";
    } else {
        # If no mapping found, map to itself
        print $output_fh "$id $id\n";
    }
}

close($phones_fh);
close($output_fh);

print STDERR "Created phone mapping: $output_file\n";